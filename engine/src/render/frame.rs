use std::error::Error;

use ash::Device;
use ash::vk;

use super::depth::DepthBuffer;
use super::descriptor::DescriptorSet;
use super::device::DeviceContext;
use super::pipeline::GraphicsPipeline;
use super::swapchain::Swapchain;

/// clear color for the background of every frame
/// i like indigo.
const CLEAR_COLOR: [f32; 4] = [0.06, 0.08, 0.16, 1.0];

/// one batched draw: all instances of one interned shape, drawn with a
/// single `vkcmddrawindexed` call; instances are addressed through the
/// uniform buffer by `first_instance..first_instance + instance_count`;
pub struct ShapeDraw {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub index_count: u32,
    pub first_instance: u32,
    pub instance_count: u32,
    pub style: crate::scene::RenderStyle,
}

/// per-swapchain resources: a render pass that clears the swapchain image and
/// depth buffer, one framebuffer per image, one command buffer per image
/// recorded per frame, plus sync objects for two frames in flight;
///
/// two fences + two acquire semaphores throttle the cpu loop (the acquire
/// semaphore must exist before the image index is known); present semaphores
/// are indexed per swapchain image so one is never re-signaled while a prior
/// present of the same image is still in use
pub struct Frame {
    pub render_pass: vk::RenderPass,
    pub ui_render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub ui_framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub extent: vk::Extent2D,
    depth: DepthBuffer,
    pub fences: [vk::Fence; 2],
    pub acquire_semaphores: [vk::Semaphore; 2],
    pub present_semaphores: Vec<vk::Semaphore>,
    pub frame_index: usize,
}

impl Frame {
    pub fn create(
        device: &DeviceContext,
        queue_family: u32,
        swapchain: &Swapchain,
    ) -> Result<Self, Box<dyn Error>> {
        let render_pass = create_render_pass(&device.device, swapchain.format.format)?;
        let ui_render_pass = create_ui_render_pass(&device.device, swapchain.format.format)?;
        let depth = DepthBuffer::create(device, swapchain.extent)?;

        let framebuffers = swapchain
            .views
            .iter()
            .map(|&view| {
                let attachments = [view, depth.handle()];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);
                unsafe { device.device.create_framebuffer(&create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // color-only framebuffers for the ui overlay pass, which loads the
        // already-rendered scene
        let ui_framebuffers = swapchain
            .views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(ui_render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);
                unsafe { device.device.create_framebuffer(&create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // reset_command_buffer: command buffers are re-recorded every frame
        let command_pool = unsafe {
            device.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        let command_buffers = unsafe {
            device.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(framebuffers.len() as u32),
            )?
        };

        // fences start signaled so the first frame's wait passes before any
        // submit has completef
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fences = [
            unsafe { device.device.create_fence(&fence_info, None)? },
            unsafe { device.device.create_fence(&fence_info, None)? },
        ];

        // acquire semaphores per in-flight slot (the semaphore must be passed
        // to acquire before the image index is known); present semaphores per
        // swapchain image so a semaphore is only reused once its image has
        // been released by the presentation engine (its prior present done)
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let acquire_semaphores = [
            unsafe { device.device.create_semaphore(&semaphore_info, None)? },
            unsafe { device.device.create_semaphore(&semaphore_info, None)? },
        ];
        let present_semaphores = (0..swapchain.views.len())
            .map(|_| unsafe { device.device.create_semaphore(&semaphore_info, None) })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            render_pass,
            ui_render_pass,
            framebuffers,
            ui_framebuffers,
            command_pool,
            command_buffers,
            extent: swapchain.extent,
            depth,
            fences,
            acquire_semaphores,
            present_semaphores,
            frame_index: 0,
        })
    }

    /// record the frame for `image_index`: clear color + depth, then draw
    /// every shape group with one instanced draw call each
    pub fn record(
        &mut self,
        device: &Device,
        image_index: usize,
        solid_pipeline: &GraphicsPipeline,
        wireframe_pipeline: &GraphicsPipeline,
        descriptors: &DescriptorSet,
        draws: &[ShapeDraw],
        push_bytes: &[u8],
    ) -> Result<(), vk::Result> {
        let command_buffer = self.command_buffers[image_index];

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        // lighting block (camera + light), shared by every draw
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                solid_pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                push_bytes,
            );
        }

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: self.extent,
        };

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: CLEAR_COLOR,
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(scissor)
            .clear_values(&clear_values);

        unsafe {
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            // the uniform buffer covers the whole scene; instances are
            // addressed by index. one bind serves every draw
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                solid_pipeline.layout,
                0,
                &[descriptors.set],
                &[],
            );

            for draw in draws {
                let pipeline = match draw.style {
                    crate::scene::RenderStyle::Solid => solid_pipeline.pipeline,
                    crate::scene::RenderStyle::Wireframe => wireframe_pipeline.pipeline,
                };
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                let vertex_buffers = [draw.vertex_buffer];
                let bind_offsets = [0u64];
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &bind_offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    draw.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                // one draw for every instance of this shape; the shader
                // picks its mvp via gl_instanceindex (= firstinstance + i)
                device.cmd_draw_indexed(
                    command_buffer,
                    draw.index_count,
                    draw.instance_count,
                    0,
                    0,
                    draw.first_instance,
                );
            }

            device.cmd_end_render_pass(command_buffer);
        }
        Ok(())
    }

    /// destroy all frame resources; the device has to outlive them
    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            for &fence in &self.fences {
                device.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.acquire_semaphores {
                device.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.present_semaphores {
                device.device.destroy_semaphore(semaphore, None);
            }
            device
                .device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            device.device.destroy_command_pool(self.command_pool, None);
            for &framebuffer in &self.ui_framebuffers {
                device.device.destroy_framebuffer(framebuffer, None);
            }
            for &framebuffer in &self.framebuffers {
                device.device.destroy_framebuffer(framebuffer, None);
            }
            device.device.destroy_render_pass(self.render_pass, None);
            device.device.destroy_render_pass(self.ui_render_pass, None);
        }
        self.depth.destroy(device);
        log::info!("destroyed frame resources");
    }
}

fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, vk::Result> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // the ui overlay pass loads this attachment afterwards, so it must
        // stay in color-attachment layout; the ui pass flips it to present
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_reference = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_reference = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_references = [color_reference];
    let subpass = vk::SubpassDescription::default()
        .color_attachments(&color_references)
        .depth_stencil_attachment(&depth_reference);

    let attachments = [color_attachment, depth_attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    unsafe { device.create_render_pass(&create_info, None) }
}

fn create_ui_render_pass(
    device: &Device,
    format: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_reference = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_references = [color_reference];
    let subpass = vk::SubpassDescription::default().color_attachments(&color_references);

    let attachments = [attachment];
    let subpasses = [subpass];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);

    unsafe { device.create_render_pass(&create_info, None) }
}
