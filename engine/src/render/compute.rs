//! compute dispatch for the gpu broad and narrow phases; rust-gpu kernels run
//! a support-based aabb sweep followed by gjk and epa per candidate pair
//! the host reads up to four contact slots per pair after a fence

use std::error::Error;
use std::mem::size_of;

use ash::vk;

use glam::{Quat, Vec3};

use super::buffer::Buffer;
use super::device::DeviceContext;
use super::shader;

/// the compiled rust-gpu narrow-phase kernel
pub const NARROWPHASE_SPIRV: &[u8] = include_bytes!("../../shaders/gpu_physics/narrowphase.spv");

/// the gpu broad-phase kernels (aabb / bitonic sort / sweep)
pub const BROADPHASE_SPIRV: &[u8] = include_bytes!("../../shaders/gpu_physics/broadphase.spv");

const WORKGROUP_SIZE: u32 = 64;
const CONTACTS_PER_PAIR: usize = 4;

/// packed shape parameters
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuShape {
    pub tag: u32,
    pub pad: [u32; 3],
    /// sphere: radius in `corners[0][0]`; cube: half-extent in `corners[0][0]`
    /// tetrahedron: the four corners
    pub corners: [[f32; 4]; 4],
}

impl GpuShape {
    pub const SPHERE: u32 = 0;
    pub const CUBE: u32 = 1;
    pub const TETRAHEDRON: u32 = 2;

    pub fn sphere(radius: f32) -> Self {
        Self {
            tag: Self::SPHERE,
            pad: [0; 3],
            corners: [[radius, 0.0, 0.0, 0.0], [0.0; 4], [0.0; 4], [0.0; 4]],
        }
    }

    pub fn cube(half_extent: f32) -> Self {
        Self {
            tag: Self::CUBE,
            pad: [0; 3],
            corners: [
                [half_extent, half_extent, half_extent, 0.0],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ],
        }
    }

    pub fn tetrahedron(corners: [Vec3; 4]) -> Self {
        let mut packed = [[0.0f32; 4]; 4];
        for (i, corner) in corners.iter().enumerate() {
            packed[i] = corner.extend(0.0).to_array();
        }
        Self {
            tag: Self::TETRAHEDRON,
            pad: [0; 3],
            corners: packed,
        }
    }
}

/// a body as the kernel sees it: world transform + packed shape
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuBody {
    pub position: [f32; 4],
    pub orientation: [f32; 4], // quaternion xyzw
    pub shape: GpuShape,
}

impl GpuBody {
    pub fn new(position: Vec3, orientation: Quat, shape: GpuShape) -> Self {
        Self {
            position: position.extend(0.0).to_array(),
            orientation: orientation.to_array(),
            shape,
        }
    }
}

/// a candidate pair (indices into the bodies buffer);
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuPair {
    pub a: u32,
    pub b: u32,
}

/// broad-phase sort key: morton code + body index
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SortKey {
    pub code: u32,
    pub index: u32,
}

/// push constants shared by the narrow and broad kernels (16 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputePush {
    pub pair_count: u32,
    pub body_count: u32,
    pub max_pairs: u32,
    pub sort_len: u32,
}

/// raw kernel output slot per pair; `valid != 0` marks a contact
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuContactOut {
    pub valid: u32,
    pub a: u32,
    pub b: u32,
    pub pad: u32,
    pub normal: [f32; 4],
    pub depth: f32,
    pub pad2: [f32; 3],
    pub point_a: [f32; 4],
    pub point_b: [f32; 4],
}

/// a valid contact, decoded for cpu consumers; `normal` points from `b`
/// toward `a`'s counterpart: it is the direction to move `b` by `depth` to
/// separate it from `a`
#[derive(Clone, Copy, Debug)]
pub struct GpuContact {
    pub a: usize,
    pub b: usize,
    pub normal: Vec3,
    pub depth: f32,
    pub point_a: Vec3,
    pub point_b: Vec3,
}

/// the compute pipeline and buffers for one narrow-phase dispatch
pub struct NarrowPhaseCompute {
    pipeline: vk::Pipeline,
    broad_pipelines: [vk::Pipeline; 2],
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    bodies: Buffer,
    pairs: Buffer,
    keys: Buffer,
    aabbs: Buffer,
    pair_count: Buffer,
    // two contact buffers alternate per dispatch so a submit can overwrite
    // one while the caller is still reading the other's results
    contacts: [Buffer; 2],
    descriptor_sets: [vk::DescriptorSet; 2],
    /// which contact buffer the most recent [`self::submit`] wrote
    buffer_index: usize,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    max_bodies: usize,
    max_pairs: usize,
    /// pair count of the most recent submit, for the readback size
    last_pairs: usize,
}

impl NarrowPhaseCompute {
    pub fn new(
        device: &DeviceContext,
        max_bodies: usize,
        max_pairs: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let module = shader::create_module(&device.device, NARROWPHASE_SPIRV, "narrowphase")?;
        let broad_module = shader::create_module(&device.device, BROADPHASE_SPIRV, "broadphase")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"narrowphase_main");

        // descriptor set: bodies(0) / pairs(1) / contacts(2) / keys(3) /
        // aabbs(4) / pair_count(5); the broad kernels use 0,1,3,4,5 and the
        // narrow kernel 0,1,2
        let make_binding = |binding: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings: Vec<_> = (0..6).map(make_binding).collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 12,
        };
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let descriptor_pool = unsafe { device.device.create_descriptor_pool(&pool_info, None)? };

        let set_layouts = [descriptor_layout, descriptor_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets = unsafe { device.device.allocate_descriptor_sets(&alloc_info)? };
        let descriptor_sets: [vk::DescriptorSet; 2] =
            descriptor_sets.try_into().expect("two descriptor sets");

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let bodies = Buffer::create(
            device,
            (max_bodies * size_of::<GpuBody>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let pairs = Buffer::create(
            device,
            (max_pairs * size_of::<GpuPair>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let keys = Buffer::create(
            device,
            (max_bodies * size_of::<SortKey>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let aabbs = Buffer::create(
            device,
            (2 * max_bodies * size_of::<[f32; 4]>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let pair_count = Buffer::create(device, size_of::<u32>() as vk::DeviceSize, storage, host)?;
        let contacts = [
            Buffer::create(
                device,
                (max_pairs * CONTACTS_PER_PAIR * size_of::<GpuContactOut>()) as vk::DeviceSize,
                storage,
                host,
            )?,
            Buffer::create(
                device,
                (max_pairs * CONTACTS_PER_PAIR * size_of::<GpuContactOut>()) as vk::DeviceSize,
                storage,
                host,
            )?,
        ];
        for (set, contact_buffer) in descriptor_sets.iter().zip(&contacts) {
            let buffer_infos = [
                vk::DescriptorBufferInfo {
                    buffer: bodies.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: pairs.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: contact_buffer.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: keys.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: aabbs.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                vk::DescriptorBufferInfo {
                    buffer: pair_count.buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..6)
                .map(|binding| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(binding)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_infos[binding as usize]))
                })
                .collect();
            unsafe { device.device.update_descriptor_sets(&writes, &[]) };
        }

        // pipeline layout with the pair-count push constant
        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<ComputePush>() as u32,
        }];
        let descriptor_layouts = [descriptor_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(&push_ranges);
        let layout = unsafe { device.device.create_pipeline_layout(&layout_info, None)? };

        let make_pipeline =
            |stage: vk::PipelineShaderStageCreateInfo| -> Result<vk::Pipeline, Box<dyn Error>> {
                let pipeline_info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout);
                Ok(unsafe {
                    device.device.create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[pipeline_info],
                        None,
                    )
                }
                .map_err(|(_, error)| error)?[0])
            };
        let pipeline = make_pipeline(stage)?;
        let broad_stages = [c"broad_aabb", c"broad_sweep_dense"].map(|name| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(broad_module)
                .name(name)
        });
        let mut broad_pipelines = [vk::Pipeline::null(); 2];
        for (i, stage) in broad_stages.into_iter().enumerate() {
            broad_pipelines[i] = make_pipeline(stage)?;
        }
        unsafe {
            device.device.destroy_shader_module(module, None);
            device.device.destroy_shader_module(broad_module, None);
        };

        let command_pool = unsafe {
            device.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command_buffer = unsafe {
            device.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        let fence = unsafe {
            device.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        Ok(Self {
            pipeline,
            broad_pipelines,
            layout,
            descriptor_layout,
            descriptor_pool,
            bodies,
            pairs,
            keys,
            aabbs,
            pair_count,
            contacts,
            descriptor_sets,
            buffer_index: 0,
            command_pool,
            command_buffer,
            fence,
            max_bodies,
            max_pairs,
            last_pairs: 0,
        })
    }

    /// submit the kernel over `pairs`; the dispatch writes into the contact
    /// buffer alternated away from the one the caller may still be reading
    /// the previous dispatch must be complete before re-recording the shared
    /// command buffer, so this waits the prior fence first (cheap: a full
    /// frame has usually elapsed)
    pub fn submit(
        &mut self,
        device: &DeviceContext,
        bodies: &[GpuBody],
        pairs: &[GpuPair],
    ) -> Result<(), Box<dyn Error>> {
        assert!(
            bodies.len() <= self.max_bodies,
            "too many bodies for the compute buffers"
        );
        assert!(
            pairs.len() <= self.max_pairs,
            "too many pairs for the compute buffers"
        );
        self.last_pairs = pairs.len();
        if pairs.is_empty() {
            return Ok(());
        }

        self.buffer_index ^= 1;
        let contact_buffer = &mut self.contacts[self.buffer_index];
        let descriptor_set = self.descriptor_sets[self.buffer_index];

        // the command buffer is reused: the previous dispatch must be done
        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.bodies.write(device, bytes_of(bodies))?;
        self.pairs.write(device, bytes_of(pairs))?;

        // clear all reserved slots, including unused manifold slots, so a
        // dispatch never sees stale contacts from an earlier one
        let zero = vec![0u8; pairs.len() * CONTACTS_PER_PAIR * size_of::<GpuContactOut>()];
        contact_buffer.write(device, &zero)?;

        unsafe {
            device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            device.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[descriptor_set],
                &[],
            );
            let pair_count = pairs.len() as u32;
            device.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pair_count.to_ne_bytes(),
            );
            let groups = pair_count.div_ceil(WORKGROUP_SIZE);
            device
                .device
                .cmd_dispatch(self.command_buffer, groups, 1, 1);
            device.device.end_command_buffer(self.command_buffer)?;

            device.device.reset_fences(&[self.fence])?;
            let command_buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .device
                .queue_submit(device.queue, &[submit], self.fence)?;
        }
        Ok(())
    }

    /// gpu broad phase: per-body aabb + morton key, bitonic sort, and the
    /// sweep that appends overlapping pairs to the pairs buffer (atomic
    /// counter); read the count with `self::read_pair_count`, then run the
    /// narrow phase with `self::submit_narrow`
    pub fn submit_broad(
        &mut self,
        device: &DeviceContext,
        bodies: &[GpuBody],
    ) -> Result<(), Box<dyn Error>> {
        assert!(
            bodies.len() <= self.max_bodies,
            "too many bodies for the compute buffers"
        );
        if bodies.is_empty() {
            return Ok(());
        }
        let n = bodies.len() as u32;

        // the command buffer is reused: the previous dispatch must be done
        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.bodies.write(device, bytes_of(bodies))?;
        self.pair_count.write(device, &[0u8; 4])?;

        let push = ComputePush {
            pair_count: 0,
            body_count: n,
            max_pairs: self.max_pairs as u32,
            sort_len: n * n,
        };

        unsafe {
            device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            device.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_sets[0]],
                &[],
            );
            device.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(std::slice::from_ref(&push)),
            );
            let groups = n.div_ceil(WORKGROUP_SIZE);
            let dispatch = |pipeline: vk::Pipeline, count: u32| {
                device.device.cmd_bind_pipeline(
                    self.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline,
                );
                device.device.cmd_dispatch(self.command_buffer, count, 1, 1);
            };
            dispatch(self.broad_pipelines[0], groups); // aabb
            // dense O(n^2) pair sweep; the bitonic sort was removed because
            // its shared-memory barriers cost ~5 ms per dispatch lol
            let pair_groups = (n * n).div_ceil(WORKGROUP_SIZE);
            dispatch(self.broad_pipelines[1], pair_groups); // sweep
            device.device.end_command_buffer(self.command_buffer)?;

            device.device.reset_fences(&[self.fence])?;
            let command_buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .device
                .queue_submit(device.queue, &[submit], self.fence)?;
        }
        Ok(())
    }

    /// wait for the broad phase and return the candidate pair count
    pub fn read_pair_count(&mut self, device: &DeviceContext) -> Result<usize, Box<dyn Error>> {
        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let bytes = self.pair_count.read(device, 4)?;
        let count = u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as usize;
        Ok(count)
    }

    /// read back `count` pairs from the pairs buffer (written by the broad
    /// phase sweep), for parity checks
    pub fn read_pairs(
        &self,
        device: &DeviceContext,
        count: usize,
    ) -> Result<Vec<GpuPair>, Box<dyn Error>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.pairs.read(device, count * size_of::<GpuPair>())?;
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(size_of::<GpuPair>()) {
            out.push(unsafe { (chunk.as_ptr() as *const GpuPair).read() });
        }
        Ok(out)
    }

    /// dispatch the narrow phase over the pairs the broad phase already
    /// wrote into the pairs buffer (bodies/pairs are not re-uploaded)
    pub fn submit_narrow(
        &mut self,
        device: &DeviceContext,
        pair_count: usize,
    ) -> Result<(), Box<dyn Error>> {
        assert!(
            pair_count <= self.max_pairs,
            "too many pairs for the compute buffers"
        );
        self.last_pairs = pair_count;
        if pair_count == 0 {
            return Ok(());
        }

        self.buffer_index ^= 1;
        let contact_buffer = &mut self.contacts[self.buffer_index];
        let descriptor_set = self.descriptor_sets[self.buffer_index];

        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        // clear the contact slots the kernel does not touch
        let zero = vec![0u8; pair_count * CONTACTS_PER_PAIR * size_of::<GpuContactOut>()];
        contact_buffer.write(device, &zero)?;

        unsafe {
            device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            device.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[descriptor_set],
                &[],
            );
            let pair_count = pair_count as u32;
            device.device.cmd_push_constants(
                self.command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pair_count.to_ne_bytes(),
            );
            let groups = pair_count.div_ceil(WORKGROUP_SIZE);
            device
                .device
                .cmd_dispatch(self.command_buffer, groups, 1, 1);
            device.device.end_command_buffer(self.command_buffer)?;

            device.device.reset_fences(&[self.fence])?;
            let command_buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .device
                .queue_submit(device.queue, &[submit], self.fence)?;
        }
        Ok(())
    }

    /// wait for the most recent [`self::submit`] and read back its raw
    /// contact slots; four slots are reserved for each candidate pair
    pub fn read(&mut self, device: &DeviceContext) -> Result<Vec<GpuContactOut>, Box<dyn Error>> {
        if self.last_pairs == 0 {
            return Ok(Vec::new());
        }
        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        let contact_buffer = &self.contacts[self.buffer_index];
        let slot_count = self.last_pairs * CONTACTS_PER_PAIR;
        let bytes = contact_buffer.read(device, slot_count * size_of::<GpuContactOut>())?;
        let mut out = Vec::with_capacity(slot_count);
        for chunk in bytes.chunks_exact(size_of::<GpuContactOut>()) {
            out.push(unsafe { (chunk.as_ptr() as *const GpuContactOut).read() });
        }
        Ok(out)
    }

    /// run the kernel over `pairs` synchronously, returning one raw slot per
    /// pair (submit plus read); useful for focused gpu checks
    pub fn run(
        &mut self,
        device: &DeviceContext,
        bodies: &[GpuBody],
        pairs: &[GpuPair],
    ) -> Result<Vec<GpuContactOut>, Box<dyn Error>> {
        self.submit(device, bodies, pairs)?;
        self.read(device)
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            device.device.destroy_fence(self.fence, None);
            device
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            device.device.destroy_command_pool(self.command_pool, None);
            device.device.destroy_pipeline(self.pipeline, None);
            for pipeline in &self.broad_pipelines {
                device.device.destroy_pipeline(*pipeline, None);
            }
            device.device.destroy_pipeline_layout(self.layout, None);
            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
        self.bodies.destroy(device);
        self.pairs.destroy(device);
        self.keys.destroy(device);
        self.aabbs.destroy(device);
        self.pair_count.destroy(device);
        for contact in &mut self.contacts {
            contact.destroy(device);
        }
    }
}

/// decode the raw kernel output back into teh valid contacts
pub fn decode_contacts(raw: &[GpuContactOut]) -> Vec<GpuContact> {
    raw.iter()
        .filter(|slot| slot.valid != 0)
        .map(|slot| GpuContact {
            a: slot.a as usize,
            b: slot.b as usize,
            normal: Vec3::from_array(slot.normal[..3].try_into().unwrap()),
            depth: slot.depth,
            point_a: Vec3::from_array(slot.point_a[..3].try_into().unwrap()),
            point_b: Vec3::from_array(slot.point_b[..3].try_into().unwrap()),
        })
        .collect()
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * size_of::<T>())
    }
}
