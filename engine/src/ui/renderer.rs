use std::error::Error;
use std::mem::size_of;

use ash::vk;
use egui::epaint::textures::TexturesDelta;
use egui::epaint::{ClippedPrimitive, Primitive};
use egui::epaint::{ColorImage, ImageData};

use super::pipeline::{UiVertex, create_pipeline};
use super::texture::{AtlasTexture, upload_image};
use crate::render::buffer::Buffer;
use crate::render::device::DeviceContext;

/// renders egui primitives into the swapchain image after the 3d scene,
/// using a screen-space pipeline with premultiplied-alpha blending
pub struct UiRenderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    atlas: Option<AtlasTexture>,
    vertex_buffer: Buffer,
    vertex_capacity: usize,
    index_buffer: Buffer,
    index_capacity: usize,
}

impl UiRenderer {
    /// create the pipeline, descriptor set and sampler; `render_pass` is the
    /// color-only ui pass created by the frame; it must be re-supplied (via
    /// `self::set_render_pass`) if the swapchain is recreated
    pub fn new(
        device: &DeviceContext,
        render_pass: vk::RenderPass,
    ) -> Result<Self, Box<dyn Error>> {
        let descriptor_layout = {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);
            let bindings = [binding];
            unsafe {
                device.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )?
            }
        };

        let descriptor_pool = {
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            };
            let pool_sizes = [pool_size];
            unsafe {
                device.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&pool_sizes)
                        .max_sets(1),
                    None,
                )?
            }
        };

        let descriptor_set = {
            let layouts = [descriptor_layout];
            unsafe {
                device.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )?[0]
            }
        };

        let sampler = {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .max_anisotropy(1.0)
                .min_lod(0.0)
                .max_lod(0.0);
            unsafe { device.device.create_sampler(&info, None)? }
        };

        let vertex_buffer = Buffer::create(
            device,
            1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let index_buffer = Buffer::create(
            device,
            1024,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let (pipeline, pipeline_layout) = create_pipeline(device, descriptor_layout, render_pass)?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_layout,
            descriptor_pool,
            descriptor_set,
            sampler,
            atlas: None,
            vertex_buffer,
            vertex_capacity: 1024 / size_of::<UiVertex>(),
            index_buffer,
            index_capacity: 1024 / size_of::<u32>(),
        })
    }

    /// recreate the pipeline for a new ui render pass (after swapchain
    /// recreation);
    pub fn set_render_pass(
        &mut self,
        device: &DeviceContext,
        render_pass: vk::RenderPass,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            device.device.destroy_pipeline(self.pipeline, None);
            device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
        let (pipeline, pipeline_layout) =
            create_pipeline(device, self.descriptor_layout, render_pass)?;
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;
        Ok(())
    }

    /// apply pending texture changes (the font atlas), then clear them
    pub fn process_textures(
        &mut self,
        device: &DeviceContext,
        delta: &mut TexturesDelta,
    ) -> Result<(), Box<dyn Error>> {
        for (_, deltas) in &delta.set {
            for image_delta in deltas {
                let ImageData::Color(image) = &image_delta.image;
                self.upload_atlas(device, image_delta.pos, image.as_ref())?;
            }
        }
        delta.clear();
        Ok(())
    }

    fn upload_atlas(
        &mut self,
        device: &DeviceContext,
        pos: Option<[usize; 2]>,
        image: &ColorImage,
    ) -> Result<(), Box<dyn Error>> {
        let width = image.width() as u32;
        let height = image.height() as u32;
        let pixels: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();

        match pos {
            // full atlas (re)placement;
            None => {
                let atlas = AtlasTexture::create(device, width, height)?;
                unsafe { device.device.device_wait_idle()? };
                upload_image(device, atlas.image, 0, 0, width, height, &pixels)?;
                if let Some(mut old) = self.atlas.take() {
                    old.destroy(device);
                }
                self.atlas = Some(atlas);
                let atlas = self.atlas.as_ref().unwrap();
                self.update_descriptor(device, atlas);
            }
            // partial update (new glyphs appended to the atlas)
            Some([x, y]) => {
                let Some(atlas) = &self.atlas else {
                    return Ok(());
                };
                upload_image(
                    device,
                    atlas.image,
                    x as u32,
                    y as u32,
                    width,
                    height,
                    &pixels,
                )?;
            }
        }
        Ok(())
    }

    fn update_descriptor(&self, device: &DeviceContext, atlas: &AtlasTexture) {
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(atlas.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);
        let writes = [write];
        unsafe { device.device.update_descriptor_sets(&writes, &[]) };
    }

    /// record the ui overlay into the frame's command buffer, after the 3d
    /// render pass; the ui pass loads the already-rendered color attachment
    pub fn record(
        &mut self,
        device: &DeviceContext,
        command_buffer: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<(), Box<dyn Error>> {
        // pack all vertices and indices into combined buffers
        let vertex_count: usize = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(mesh) => mesh.vertices.len(),
                Primitive::Callback(_) => 0,
            })
            .sum();
        let index_count: usize = primitives
            .iter()
            .map(|p| match &p.primitive {
                Primitive::Mesh(mesh) => mesh.indices.len(),
                Primitive::Callback(_) => 0,
            })
            .sum();
        if vertex_count > self.vertex_capacity {
            self.grow_vertices(device, vertex_count)?;
        }
        if index_count > self.index_capacity {
            self.grow_indices(device, index_count)?;
        }

        let mut vertices: Vec<UiVertex> = Vec::with_capacity(vertex_count);
        let mut indices: Vec<u32> = Vec::with_capacity(index_count);
        for primitive in primitives {
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                continue;
            };
            let base = vertices.len() as u32;
            vertices.extend(mesh.vertices.iter().map(|v| UiVertex {
                pos: [v.pos.x * pixels_per_point, v.pos.y * pixels_per_point],
                uv: [v.uv.x, v.uv.y],
                color: v.color.to_array(),
            }));
            indices.extend(mesh.indices.iter().map(|i| i + base));
        }

        {
            let vertex_bytes = bytes_of(&vertices);
            self.vertex_buffer.write(device, vertex_bytes)?;
            let index_bytes = bytes_of(&indices);
            self.index_buffer.write(device, index_bytes)?;
        }

        let render_area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            device.device.cmd_begin_render_pass(
                command_buffer,
                &pass_info,
                vk::SubpassContents::INLINE,
            );
            device
                .device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            device
                .device
                .cmd_set_scissor(command_buffer, 0, &[render_area]);
            device.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &screen_bytes(extent),
            );
            device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            let vertex_buffers = [self.vertex_buffer.buffer];
            let bind_offsets = [0u64];
            device.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &vertex_buffers,
                &bind_offsets,
            );
            device.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );
        }

        let mut first_index = 0u32;
        for primitive in primitives {
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                log::warn!("ignoring egui paint callback");
                continue;
            };
            let index_count = mesh.indices.len() as u32;

            // scissor from the clip rect (points -> physical pixels)
            let rect = primitive.clip_rect * pixels_per_point;
            if !rect.is_positive() {
                first_index += index_count;
                continue;
            }
            let x = rect.min.x.clamp(0.0, extent.width as f32) as i32;
            let y = rect.min.y.clamp(0.0, extent.height as f32) as i32;
            let w = (rect.max.x.clamp(0.0, extent.width as f32) - x as f32) as u32;
            let h = (rect.max.y.clamp(0.0, extent.height as f32) - y as f32) as u32;
            if w == 0 || h == 0 {
                first_index += index_count;
                continue;
            }
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D {
                    width: w,
                    height: h,
                },
            };

            unsafe {
                device.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                device
                    .device
                    .cmd_draw_indexed(command_buffer, index_count, 1, first_index, 0, 0);
            }
            first_index += index_count;
        }

        unsafe {
            device.device.cmd_end_render_pass(command_buffer);
        }
        Ok(())
    }

    fn grow_vertices(&mut self, device: &DeviceContext, count: usize) -> Result<(), vk::Result> {
        unsafe { device.device.device_wait_idle().map_err(|e| e)? };
        let capacity = count.next_power_of_two();
        let new = Buffer::create(
            device,
            (capacity * size_of::<UiVertex>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|_| vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let mut old = std::mem::replace(&mut self.vertex_buffer, new);
        old.destroy(device);
        self.vertex_capacity = capacity;
        Ok(())
    }

    fn grow_indices(&mut self, device: &DeviceContext, count: usize) -> Result<(), vk::Result> {
        unsafe { device.device.device_wait_idle().map_err(|e| e)? };
        let capacity = count.next_power_of_two();
        let new = Buffer::create(
            device,
            (capacity * size_of::<u32>()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(|_| vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
        let mut old = std::mem::replace(&mut self.index_buffer, new);
        old.destroy(device);
        self.index_capacity = capacity;
        Ok(())
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            device.device.destroy_pipeline(self.pipeline, None);
            device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.device.destroy_sampler(self.sampler, None);
        }
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        if let Some(mut atlas) = self.atlas.take() {
            atlas.destroy(device);
        }
        log::info!("destroyed ui renderer");
    }
}

fn bytes_of<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of::<T>() * data.len()) }
}

fn screen_bytes(extent: vk::Extent2D) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&(extent.width as f32).to_ne_bytes());
    bytes[4..].copy_from_slice(&(extent.height as f32).to_ne_bytes());
    bytes
}
