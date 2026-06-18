//! gpu avbd solver: the block sweeps of `physics::gpu` as rust-gpu compute
//! kernels (`shaders/gpu_physics/avbd;spv`, four entry points); the host drives
//! the iteration loop with one command buffer per step

use ash::vk;
use glam::{Quat, Vec3, Vec4};

use super::buffer::Buffer;
use super::device::DeviceContext;
use super::shader;

/// embedded spir-v: the avbd compute kernels (init, primal, dual, recover)
pub const AVBD_SPIRV: &[u8] = include_bytes!("../../shaders/gpu_physics/avbd.spv");

/// per-body dynamics state, c.f. `avbd/src/lib;rs::gpuavbdbodystate`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdBodyState {
    pub vel: Vec4,
    pub ang: Vec4,
    pub prev_vel: Vec4,
    pub inv_moment: Vec4,
    pub inv_mass: f32,
    pub friction: f32,
    pub _pad: [f32; 2],
}

/// one contact, c.f. `avbd/src/lib;rs::gpuavbdcontact`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdContact {
    pub a: u32,
    pub b: u32,
    pub friction: f32,
    pub _pad: f32,
    pub normal: Vec4,
    pub tangent1: Vec4,
    pub tangent2: Vec4,
    pub r_a: Vec4,
    pub r_b: Vec4,
    pub c0: Vec4,
    pub penalty: Vec4,
    pub lambda: Vec4,
}

/// constraint record, matching the gpu shader's constraint storage
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuAvbdConstraint {
    pub a: u32,
    pub b: u32,
    pub kind: u32,
    pub _pad: u32,
    pub anchor_a: Vec4,
    pub anchor_b: Vec4,
}

/// push constants, matching the shader's `push` (48 bytes);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvbdPush {
    pub dt: f32,
    pub alpha: f32,
    pub beta_lin: f32,
    pub penalty_max: f32,
    pub gravity: [f32; 4],
    pub body_count: u32,
    pub contact_count: u32,
    pub constraint_count: u32,
    pub parity: u32,
}

/// tuning knobs for one gpu step (mirror of the solver's `avbdoptions`
/// subset the kernels need);
#[derive(Clone, Copy)]
pub struct AvbdRunOptions {
    pub dt: f32,
    pub gravity: Vec3,
    pub alpha: f32,
    pub beta_lin: f32,
    pub penalty_max: f32,
    pub iterations: u32,
}

/// the state the kernels read back after a step;
pub struct AvbdGpuResult {
    pub positions: Vec<Vec3>,
    pub orientations: Vec<Quat>,
    pub velocities: Vec<Vec3>,
    pub angular_velocities: Vec<Vec3>,
    pub prev_velocities: Vec<Vec3>,
    pub lambda: Vec<Vec3>,
    pub penalty: Vec<Vec3>,
}

pub const CONTAINER: u32 = u32::MAX;
pub const CONSTRAINT_SPRING: u32 = 0;
pub const CONSTRAINT_ROD: u32 = 1;
pub const CONSTRAINT_BALL_SOCKET: u32 = 2;
pub const CONSTRAINT_ROPE: u32 = 3;

const WORKGROUP_SIZE: u32 = 64;

/// serialize solver bodies into the kernel's state layout (positions and
/// orientations go to the alternating pos/rot buffers separately);
pub fn state_from_bodies(bodies: &[crate::physics::gpu::AvbdBody]) -> Vec<GpuAvbdBodyState> {
    bodies
        .iter()
        .map(|b| GpuAvbdBodyState {
            vel: b.velocity.extend(0.0),
            ang: b.angular_velocity.extend(0.0),
            prev_vel: b.prev_velocity.extend(0.0),
            inv_moment: b.inv_moment.extend(0.0),
            inv_mass: b.inv_mass,
            friction: b.friction,
            _pad: [0.0; 2],
        })
        .collect()
}

/// serialize finalized contacts (warmstarted lambda / penalty included) into the
/// kernel's layout
/// `container` bodies map to u32::max
pub fn contacts_from_avbd(contacts: &[crate::physics::gpu::AvbdContact]) -> Vec<GpuAvbdContact> {
    contacts
        .iter()
        .map(|c| GpuAvbdContact {
            a: c.a as u32,
            b: if c.b == usize::MAX {
                CONTAINER
            } else {
                c.b as u32
            },
            friction: c.friction,
            _pad: 0.0,
            normal: c.normal.extend(0.0),
            tangent1: c.tangent1.extend(0.0),
            tangent2: c.tangent2.extend(0.0),
            r_a: c.r_a.extend(0.0),
            r_b: c.r_b.extend(0.0),
            c0: c.c0.extend(0.0),
            penalty: c.penalty.extend(0.0),
            lambda: c.lambda.extend(0.0),
        })
        .collect()
}

/// serialize springs, rods, ball-and-socket joints and ropes
pub fn constraints_from_avbd(constraints: &[crate::physics::Constraint]) -> Vec<GpuAvbdConstraint> {
    constraints
        .iter()
        .map(|constraint| {
            let (a, b, kind, anchor_a, anchor_b) = match constraint {
                crate::physics::Constraint::Spring(c) => (
                    c.body_a,
                    c.body_b,
                    CONSTRAINT_SPRING,
                    c.anchor_a.extend(c.rest_length),
                    c.anchor_b.extend(c.stiffness),
                ),
                crate::physics::Constraint::Rod(c) => (
                    c.body_a,
                    c.body_b,
                    CONSTRAINT_ROD,
                    c.anchor_a.extend(c.rest_length),
                    c.anchor_b.extend(c.stiffness),
                ),
                crate::physics::Constraint::BallSocket(c) => (
                    c.body_a,
                    c.body_b,
                    CONSTRAINT_BALL_SOCKET,
                    c.anchor_a.extend(0.0),
                    c.anchor_b.extend(c.stiffness),
                ),
                crate::physics::Constraint::Rope(c) => (
                    c.body_a,
                    c.body_b,
                    CONSTRAINT_ROPE,
                    c.anchor_a.extend(c.max_length),
                    c.anchor_b.extend(c.stiffness),
                ),
            };
            GpuAvbdConstraint {
                a: a as u32,
                b: b as u32,
                kind,
                _pad: 0,
                anchor_a,
                anchor_b,
            }
        })
        .collect()
}

/// the compute pipeline, buffers and per-step dispatch for the avbd solver
pub struct AvbdCompute {
    pipelines: [vk::Pipeline; 4],
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    state: Buffer,
    contacts: Buffer,
    constraints: Buffer,
    offsets: Buffer,
    indices: Buffer,
    initial: Buffer,
    inertial: Buffer,
    pos: [Buffer; 2],
    rot: [Buffer; 2],
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    max_bodies: usize,
    max_contacts: usize,
    max_indices: usize,
    /// state sizes of the most recent submit, for the readback
    last_n: usize,
    last_m: usize,
    last_iterations: u32,
}

impl AvbdCompute {
    pub fn new(
        device: &DeviceContext,
        max_bodies: usize,
        max_contacts: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let module = shader::create_module(&device.device, AVBD_SPIRV, "avbd")?;
        const ENTRIES: [&std::ffi::CStr; 4] =
            [c"avbd_init", c"avbd_primal", c"avbd_dual", c"avbd_recover"];

        // descriptor set: 10 storage buffers, bindings 0..=9;
        // binding 10 is reserved for the next constraint family
        let make_binding = |binding: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings: Vec<_> = (0..11).map(make_binding).collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_layout = unsafe {
            device
                .device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 11,
        };
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.device.create_descriptor_pool(&pool_info, None)? };

        let set_layouts = [descriptor_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.device.allocate_descriptor_sets(&alloc_info)? }[0];

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let max_indices = 2 * max_contacts;
        let state = Buffer::create(
            device,
            (max_bodies * size_of::<GpuAvbdBodyState>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let contacts = Buffer::create(
            device,
            (max_contacts * size_of::<GpuAvbdContact>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let constraints = Buffer::create(
            device,
            (max_contacts * size_of::<GpuAvbdConstraint>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let offsets = Buffer::create(
            device,
            ((max_bodies + 1) * size_of::<u32>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let indices = Buffer::create(
            device,
            (max_indices * size_of::<u32>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let initial = Buffer::create(
            device,
            (2 * max_bodies * size_of::<Vec4>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let inertial = Buffer::create(
            device,
            (2 * max_bodies * size_of::<Vec4>()) as vk::DeviceSize,
            storage,
            host,
        )?;
        let mk = |size: usize| Buffer::create(device, size as vk::DeviceSize, storage, host);
        let pos = [
            mk(max_bodies * size_of::<Vec4>())?,
            mk(max_bodies * size_of::<Vec4>())?,
        ];
        let rot = [
            mk(max_bodies * size_of::<Vec4>())?,
            mk(max_bodies * size_of::<Vec4>())?,
        ];

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = [
            &state,
            &contacts,
            &offsets,
            &indices,
            &initial,
            &inertial,
            &pos[0],
            &rot[0],
            &pos[1],
            &rot[1],
            &constraints,
        ]
        .iter()
        .map(|buffer| vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..11)
            .map(|binding| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[binding as usize]))
            })
            .collect();
        unsafe { device.device.update_descriptor_sets(&writes, &[]) };

        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<AvbdPush>() as u32,
        }];
        let descriptor_layouts = [descriptor_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_layouts)
            .push_constant_ranges(&push_ranges);
        let layout = unsafe { device.device.create_pipeline_layout(&layout_info, None)? };

        let mut pipelines = [vk::Pipeline::null(); 4];
        for (i, entry) in ENTRIES.iter().enumerate() {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipeline = unsafe {
                device.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
            }
            .map_err(|(_, error)| error)?[0];
            pipelines[i] = pipeline;
        }
        unsafe { device.device.destroy_shader_module(module, None) };

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
            pipelines,
            layout,
            descriptor_layout,
            descriptor_pool,
            descriptor_set,
            state,
            contacts,
            constraints,
            offsets,
            indices,
            initial,
            inertial,
            pos,
            rot,
            command_pool,
            command_buffer,
            fence,
            max_bodies,
            max_contacts,
            max_indices,
            last_n: 0,
            last_m: 0,
            last_iterations: 0,
        })
    }

    /// submit one avbd step: upload the bodies/contacts/csr and drive the
    /// iteration loop without waiting; read the result with [`self::read`]
    /// after a frame has elapsed
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        device: &DeviceContext,
        body_state: &[GpuAvbdBodyState],
        positions: &[Vec3],
        orientations: &[Quat],
        contacts: &[GpuAvbdContact],
        constraints: &[GpuAvbdConstraint],
        offsets: &[u32],
        indices: &[u32],
        options: &AvbdRunOptions,
    ) -> Result<(), Box<dyn Error>> {
        let n = body_state.len();
        let m = contacts.len();
        assert!(n <= self.max_bodies, "too many bodies for the AVBD buffers");
        assert!(
            m <= self.max_contacts,
            "too many contacts for the AVBD buffers"
        );
        assert!(indices.len() <= self.max_indices, "too many CSR indices");
        self.last_n = n;
        self.last_m = m;
        self.last_iterations = options.iterations;
        if n == 0 {
            return Ok(());
        }

        // upload: state + the iteration-0 snapshot (positions/orientations)
        self.state.write(device, bytes_of(body_state))?;
        self.contacts.write(device, bytes_of(contacts))?;
        self.constraints.write(device, bytes_of(constraints))?;
        self.offsets.write(device, bytes_of(offsets))?;
        self.indices.write(device, bytes_of(indices))?;
        let pos0: Vec<Vec4> = positions.iter().map(|p| p.extend(0.0)).collect();
        let rot0: Vec<Vec4> = orientations
            .iter()
            .map(|q| Vec4::new(q.x, q.y, q.z, q.w))
            .collect();
        self.pos[0].write(device, bytes_of(&pos0))?;
        self.rot[0].write(device, bytes_of(&rot0))?;
        // seed both parity buffers; static bodies are not written by primal,
        // so their snapshots must remain valid on either parity
        self.pos[1].write(device, bytes_of(&pos0))?;
        self.rot[1].write(device, bytes_of(&rot0))?;

        let push = AvbdPush {
            dt: options.dt,
            alpha: options.alpha,
            beta_lin: options.beta_lin,
            penalty_max: options.penalty_max,
            gravity: {
                let g = options.gravity;
                [g.x, g.y, g.z, 0.0]
            },
            body_count: n as u32,
            contact_count: m as u32,
            constraint_count: constraints.len() as u32,
            parity: 0,
        };
        let final_parity = (options.iterations - 1) % 2;

        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
            device
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?;

            let dispatch = |cb, pipeline: vk::Pipeline, groups: u32| {
                device
                    .device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                device.device.cmd_dispatch(cb, groups.max(1), 1, 1);
            };
            let push_constants = |cb, parity: u32| {
                let mut p = push;
                p.parity = parity;
                device.device.cmd_push_constants(
                    cb,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(std::slice::from_ref(&p)),
                );
            };

            let body_groups = n.div_ceil(WORKGROUP_SIZE as usize) as u32;
            let contact_groups = m.div_ceil(WORKGROUP_SIZE as usize) as u32;

            push_constants(self.command_buffer, 0);
            dispatch(self.command_buffer, self.pipelines[0], body_groups);
            for k in 0..options.iterations {
                let parity = k % 2;
                push_constants(self.command_buffer, parity);
                dispatch(self.command_buffer, self.pipelines[1], body_groups);
                dispatch(self.command_buffer, self.pipelines[2], contact_groups);
            }
            push_constants(self.command_buffer, final_parity);
            dispatch(self.command_buffer, self.pipelines[3], body_groups);

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

    /// wait for the most recent 1self::submit` and read back the final
    /// state (positions, velocities, contact multipliers)
    pub fn read(&mut self, device: &DeviceContext) -> Result<AvbdGpuResult, Box<dyn Error>> {
        let n = self.last_n;
        let m = self.last_m;
        if n == 0 {
            return Ok(AvbdGpuResult {
                positions: Vec::new(),
                orientations: Vec::new(),
                velocities: Vec::new(),
                angular_velocities: Vec::new(),
                prev_velocities: Vec::new(),
                lambda: Vec::new(),
                penalty: Vec::new(),
            });
        }
        unsafe {
            device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        let final_parity = self.last_iterations.saturating_sub(1) % 2;
        let write_target = 1 - (final_parity as usize);
        let bytes_pos = self.pos[write_target].read(device, n * size_of::<Vec4>())?;
        let bytes_rot = self.rot[write_target].read(device, n * size_of::<Vec4>())?;
        let state_bytes = self.state.read(device, n * size_of::<GpuAvbdBodyState>())?;
        let contact_bytes = self
            .contacts
            .read(device, m * size_of::<GpuAvbdContact>())?;

        let positions: Vec<Vec3> = bytes_pos
            .chunks_exact(size_of::<Vec4>())
            .map(|chunk| unsafe { (chunk.as_ptr() as *const Vec4).read() }.truncate())
            .collect();
        let orientations: Vec<Quat> = bytes_rot
            .chunks_exact(size_of::<Vec4>())
            .map(|chunk| {
                let v = unsafe { (chunk.as_ptr() as *const Vec4).read() };
                Quat::from_xyzw(v.x, v.y, v.z, v.w)
            })
            .collect();
        let states: Vec<GpuAvbdBodyState> = state_bytes
            .chunks_exact(size_of::<GpuAvbdBodyState>())
            .map(|chunk| unsafe { (chunk.as_ptr() as *const GpuAvbdBodyState).read() })
            .collect();
        let out_contacts: Vec<GpuAvbdContact> = contact_bytes
            .chunks_exact(size_of::<GpuAvbdContact>())
            .map(|chunk| unsafe { (chunk.as_ptr() as *const GpuAvbdContact).read() })
            .collect();

        Ok(AvbdGpuResult {
            positions,
            orientations,
            velocities: states.iter().map(|s| s.vel.truncate()).collect(),
            angular_velocities: states.iter().map(|s| s.ang.truncate()).collect(),
            prev_velocities: states.iter().map(|s| s.prev_vel.truncate()).collect(),
            lambda: out_contacts.iter().map(|c| c.lambda.truncate()).collect(),
            penalty: out_contacts.iter().map(|c| c.penalty.truncate()).collect(),
        })
    }

    /// run one avbd step synchronously (submit + read), for parity checks;
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        device: &DeviceContext,
        body_state: &[GpuAvbdBodyState],
        positions: &[Vec3],
        orientations: &[Quat],
        contacts: &[GpuAvbdContact],
        constraints: &[GpuAvbdConstraint],
        offsets: &[u32],
        indices: &[u32],
        options: &AvbdRunOptions,
    ) -> Result<AvbdGpuResult, Box<dyn Error>> {
        self.submit(
            device,
            body_state,
            positions,
            orientations,
            contacts,
            constraints,
            offsets,
            indices,
            options,
        )?;
        self.read(device)
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            device.device.destroy_fence(self.fence, None);
            device
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            device.device.destroy_command_pool(self.command_pool, None);
            for pipeline in &self.pipelines {
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
        self.state.destroy(device);
        self.contacts.destroy(device);
        self.constraints.destroy(device);
        self.offsets.destroy(device);
        self.indices.destroy(device);
        self.initial.destroy(device);
        self.inertial.destroy(device);
        for buffer in &mut self.pos {
            buffer.destroy(device);
        }
        for buffer in &mut self.rot {
            buffer.destroy(device);
        }
    }
}

use std::error::Error;

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * size_of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::physics::{BallSocket, Constraint, Rod, Rope, Spring};

    #[test]
    fn constraint_records_keep_shader_layout_and_kinds() {
        let constraints = [
            Constraint::Spring(Spring::new(0, 1, Vec3::ZERO, Vec3::ZERO, 2.0, 1.0)),
            Constraint::Rod(Rod::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0)),
            Constraint::BallSocket(BallSocket::new(0, 1, Vec3::ZERO, Vec3::ZERO)),
            Constraint::Rope(Rope::new(0, 1, Vec3::ZERO, Vec3::ZERO, 3.0)),
        ];
        let records = constraints_from_avbd(&constraints);
        assert_eq!(size_of::<GpuAvbdConstraint>(), size_of::<[f32; 12]>());
        assert_eq!(
            records.iter().map(|record| record.kind).collect::<Vec<_>>(),
            [
                CONSTRAINT_SPRING,
                CONSTRAINT_ROD,
                CONSTRAINT_BALL_SOCKET,
                CONSTRAINT_ROPE
            ]
        );
        assert_eq!(records[3].anchor_a.w, 3.0);
    }
}
