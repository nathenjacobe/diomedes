use std::error::Error;

use ash::vk;
use glam::{Mat4, Vec3};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::descriptor::DescriptorSet;
use super::device::DeviceContext;
use super::frame::{Frame, ShapeDraw};
use super::instance::InstanceContext;
use super::light::Light;
use super::pipeline::GraphicsPipeline;
use super::swapchain::Swapchain;
use super::uniform::UniformBuffer;
use crate::ui::{UiFrame, UiRenderer};

/// maximum scene instances; must match the `mvp[256]`/`model[256]` arrays in
/// mesh.vert; each entry is two matrices
/// the allocated buffer is also capped by the device's uniform buffer range
const MAX_INSTANCES: usize = 256;

/// vulkan context: instance, attached surface, device, swapchain, frame
/// resources, graphics pipeline, descriptor set and uniform buffer; presents
/// a frame (clear + all mesh draws) on every redraw;
///
/// drop order is child-first via explicit teardown: frame resources and
/// pipeline, then swapchain, then device, then surface, then instance;
pub struct VulkanContext {
    frame: Option<Frame>,
    pipeline: Option<GraphicsPipeline>,
    wireframe_pipeline: Option<GraphicsPipeline>,
    swapchain: Option<Swapchain>,
    swapchain_fn: Option<ash::khr::swapchain::Device>,
    device: Option<DeviceContext>,
    surface: Option<vk::SurfaceKHR>,
    uniforms: Option<UniformBuffer>,
    descriptors: Option<DescriptorSet>,
    ui: Option<UiRenderer>,
    instance: InstanceContext,
    window_size: PhysicalSize<u32>,
}

impl VulkanContext {
    pub fn new(
        display: winit::raw_window_handle::RawDisplayHandle,
    ) -> Result<Self, Box<dyn Error>> {
        let instance = InstanceContext::new(display)?;

        Ok(Self {
            frame: None,
            pipeline: None,
            wireframe_pipeline: None,
            swapchain: None,
            swapchain_fn: None,
            device: None,
            surface: None,
            uniforms: None,
            descriptors: None,
            ui: None,
            instance,
            window_size: PhysicalSize::new(1, 1),
        })
    }

    /// whether the full draw path exists: device, swapchain, frame, pipeline,
    /// descriptors and uniform buffer
    pub fn ready(&self) -> bool {
        self.device.is_some()
            && self.swapchain.is_some()
            && self.frame.is_some()
            && self.pipeline.is_some()
            && self.descriptors.is_some()
            && self.uniforms.is_some()
    }

    /// instance slots the uniform buffer can hold on this device
    pub(crate) fn instance_capacity(&self) -> usize {
        self.uniforms
            .as_ref()
            .map_or(0, |uniforms| uniforms.capacity)
    }

    pub fn device(&self) -> Option<&DeviceContext> {
        self.device.as_ref()
    }

    /// current swapchain extent in pixels
    pub fn extent(&self) -> vk::Extent2D {
        self.frame
            .as_ref()
            .map(|frame| frame.extent)
            .unwrap_or(vk::Extent2D {
                width: 1,
                height: 1,
            })
    }

    /// attach the given window to the vulkan instance by creating a surface
    pub fn attach(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        if self.surface.is_some() {
            return Ok(());
        }

        let surface = self.instance.create_surface(window)?;
        log::info!("attached surface to window: {surface:?}");
        self.surface = Some(surface);
        Ok(())
    }

    /// create the device, swapchain, frame, pipeline, descriptor set and
    /// uniform buffer for the attached surface
    pub fn prepare(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let surface = self.surface.ok_or("no surface attached")?;
        self.window_size = window.inner_size();

        let device = DeviceContext::create(
            self.instance.instance(),
            self.instance.surface_fn(),
            surface,
        )?;
        let swapchain_fn =
            ash::khr::swapchain::Device::new(self.instance.instance(), &device.device);
        let swapchain = Swapchain::create(
            &device.device,
            &swapchain_fn,
            self.instance.surface_fn(),
            device.physical,
            surface,
            self.window_size,
        )?;
        let frame = Frame::create(&device, device.queue_family, &swapchain)?;

        let descriptors = DescriptorSet::create(&device.device)?;
        let max_uniform_slots = (device.properties.limits.max_uniform_buffer_range as usize
            / (2 * std::mem::size_of::<Mat4>()))
        .min(MAX_INSTANCES);
        log::info!(
            "instance capacity: {} (uniform range {})",
            max_uniform_slots,
            device.properties.limits.max_uniform_buffer_range
        );
        let uniforms = UniformBuffer::create(&device, max_uniform_slots)?;
        descriptors.update(&device.device, uniforms.handle());
        let pipeline = GraphicsPipeline::create(
            &device.device,
            frame.render_pass,
            descriptors.layout,
            crate::scene::RenderStyle::Solid,
        )?;
        let wireframe_pipeline = GraphicsPipeline::create(
            &device.device,
            frame.render_pass,
            descriptors.layout,
            crate::scene::RenderStyle::Wireframe,
        )?;
        let ui = UiRenderer::new(&device, frame.ui_render_pass)?;

        self.device = Some(device);
        self.swapchain_fn = Some(swapchain_fn);
        self.swapchain = Some(swapchain);
        self.frame = Some(frame);
        self.descriptors = Some(descriptors);
        self.uniforms = Some(uniforms);
        self.pipeline = Some(pipeline);
        self.wireframe_pipeline = Some(wireframe_pipeline);
        self.ui = Some(ui);
        Ok(())
    }

    /// present one frame: upload the per-instance mvp matrices, acquire the
    /// next swapchain image, record the clear + batched draws, submit and
    /// present
    pub fn render_frame(
        &mut self,
        view: &Mat4,
        projection: &Mat4,
        models: &[Mat4],
        draws: &[ShapeDraw],
        mut ui: Option<&mut UiFrame>,
        camera_position: Vec3,
        light: &Light,
    ) -> Result<(), Box<dyn Error>> {
        if !self.ready() {
            return Ok(());
        }

        // per-instance uniform data: interleaved (mvp, model) pairs, so the
        // shader can reconstruct world space for lighting
        let mut uniform_data: Vec<Mat4> = Vec::with_capacity(2 * models.len());
        for model in models {
            uniform_data.push(projection * view * model);
            uniform_data.push(*model);
        }

        {
            let uniforms = self.uniforms.as_mut().unwrap();
            let device_ctx = self.device.as_ref().unwrap();
            uniforms.write(device_ctx, &uniform_data)?;
        }

        let mut out_of_date = false;
        let image_index = {
            let device_ctx = self.device.as_ref().unwrap();
            let swapchain_obj = self.swapchain.as_ref().unwrap();
            let swapchain_fn = self.swapchain_fn.as_ref().unwrap();
            let frame = self.frame.as_mut().unwrap();
            let slot = frame.frame_index;

            unsafe {
                device_ctx
                    .device
                    .wait_for_fences(&[frame.fences[slot]], true, u64::MAX)?;
            }

            match unsafe {
                swapchain_fn.acquire_next_image(
                    swapchain_obj.swapchain,
                    u64::MAX,
                    frame.acquire_semaphores[slot],
                    vk::Fence::null(),
                )
            } {
                Ok((index, _suboptimal)) => {
                    unsafe { device_ctx.device.reset_fences(&[frame.fences[slot]])? };
                    index
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    out_of_date = true;
                    0
                }
                Err(error) => return Err(error.into()),
            }
        };

        if out_of_date {
            self.recreate_swapchain()?;
            return Ok(());
        }

        let present_result = {
            let device_ctx = self.device.as_ref().unwrap();
            let swapchain_fn = self.swapchain_fn.as_ref().unwrap();
            let solid_pipeline = self.pipeline.as_ref().unwrap();
            let wireframe_pipeline = self.wireframe_pipeline.as_ref().unwrap();
            let descriptors = self.descriptors.as_ref().unwrap();
            let frame = self.frame.as_mut().unwrap();
            let slot = frame.frame_index;
            let image = image_index as usize;

            // lighting push constants: camera + light, one set per frame
            let push_bytes = lighting_push_bytes(camera_position, light);
            frame.record(
                &device_ctx.device,
                image,
                solid_pipeline,
                wireframe_pipeline,
                descriptors,
                draws,
                &push_bytes,
            )?;

            let ui_renderer = self.ui.as_mut().unwrap();
            if let Some(ui_frame) = ui.as_deref_mut() {
                ui_renderer.process_textures(device_ctx, &mut ui_frame.textures_delta)?;
            }
            let (primitives, pixels_per_point) = match ui {
                Some(frame) => (frame.primitives.as_slice(), frame.pixels_per_point),
                None => (&[][..], 1.0),
            };
            // always run the ui pass (even with no primitives yet); its
            // final layout transition is what moves the color attachment
            // into present_src_khr
            ui_renderer.record(
                device_ctx,
                frame.command_buffers[image],
                frame.ui_render_pass,
                frame.ui_framebuffers[image],
                frame.extent,
                primitives,
                pixels_per_point,
            )?;

            unsafe {
                device_ctx
                    .device
                    .end_command_buffer(frame.command_buffers[image])?
            };

            let wait_semaphores = [frame.acquire_semaphores[slot]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [frame.command_buffers[image]];
            let signal_semaphores = [frame.present_semaphores[image]];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                device_ctx
                    .device
                    .queue_submit(device_ctx.queue, &[submit], frame.fences[slot])?;
            }

            let wait_semaphores = [frame.present_semaphores[image]];
            let swapchains = [self.swapchain.as_ref().unwrap().swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let result = unsafe { swapchain_fn.queue_present(device_ctx.queue, &present_info) };
            frame.frame_index = (slot + 1) % 2;
            result
        };

        match present_result {
            Ok(_suboptimal) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_SURFACE_LOST_KHR) => {
                self.recreate_swapchain()
            }
            Err(error) => Err(error.into()),
        }
    }

    /// recreate the swapchain and frame resources, e.g. after a resize; the
    /// pipeline uses a dynamic viewport and the uniform buffer is
    /// device-scoped, so both survive resizes untouched
    pub fn on_resized(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return; // minimized or not yet mapped
        }
        log::info!("window resized to {size:?}");
        self.window_size = size;
        if let Err(error) = self.recreate_swapchain() {
            log::error!("failed to recreate swapchain: {error}");
        }
    }

    /// destroy the surface, detaching the window from the instance, along
    /// with all device, swapchain, frame, pipeline and buffer resources
    pub fn detach(&mut self) {
        self.teardown_graphics();
        if let Some(surface) = self.surface.take() {
            self.instance.destroy_surface(surface);
            log::info!("destroyed surface");
        }
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        let device_ctx = self.device.as_ref().ok_or("no device")?;
        let surface = self.surface.ok_or("no surface")?;
        let swapchain_fn = self.swapchain_fn.as_ref().ok_or("no swapchain extension")?;

        unsafe { device_ctx.device.device_wait_idle()? };

        if let Some(mut frame) = self.frame.take() {
            frame.destroy(device_ctx);
        }
        if let Some(mut swapchain) = self.swapchain.take() {
            swapchain.destroy(&device_ctx.device, swapchain_fn);
        }

        let swapchain = Swapchain::create(
            &device_ctx.device,
            swapchain_fn,
            self.instance.surface_fn(),
            device_ctx.physical,
            surface,
            self.window_size,
        )?;
        let frame = Frame::create(device_ctx, device_ctx.queue_family, &swapchain)?;

        self.swapchain = Some(swapchain);
        self.frame = Some(frame);
        if let Some(ui) = &mut self.ui {
            ui.set_render_pass(device_ctx, self.frame.as_ref().unwrap().ui_render_pass)?;
        }
        Ok(())
    }

    fn teardown_graphics(&mut self) {
        if let Some(device_ctx) = &self.device {
            // let any in-flight frame finish before its resources go away
            unsafe {
                let _ = device_ctx.device.device_wait_idle();
            }
            if let Some(mut frame) = self.frame.take() {
                frame.destroy(device_ctx);
            }
            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.destroy(&device_ctx.device);
            }
            if let Some(mut pipeline) = self.wireframe_pipeline.take() {
                pipeline.destroy(&device_ctx.device);
            }
            if let Some(mut descriptors) = self.descriptors.take() {
                descriptors.destroy(&device_ctx.device);
            }
            if let Some(mut uniforms) = self.uniforms.take() {
                uniforms.destroy(device_ctx);
            }
            if let Some(mut ui) = self.ui.take() {
                ui.destroy(device_ctx);
            }
            if let Some(mut swapchain) = self.swapchain.take() {
                if let Some(swapchain_fn) = &self.swapchain_fn {
                    swapchain.destroy(&device_ctx.device, swapchain_fn);
                }
            }
        }
        self.swapchain_fn = None;
        self.device = None;
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.detach();
    }
}

fn lighting_push_bytes(camera_position: Vec3, light: &Light) -> [u8; 64] {
    let values = [
        camera_position.x,
        camera_position.y,
        camera_position.z,
        0.0,
        light.direction.x,
        light.direction.y,
        light.direction.z,
        0.0,
        light.color.x,
        light.color.y,
        light.color.z,
        0.0,
        light.ambient,
        light.specular_power,
        light.specular_strength,
        0.0,
    ];
    let mut bytes = [0u8; 64];
    for (i, value) in values.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
