use std::error::Error;

use ash::Device;
use ash::khr::swapchain;
use ash::vk;
use winit::dpi::PhysicalSize;

/// swapchain for presenting cleared frames to the attached surface
/// owns the swapchain handle and one image view per swapchain image
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub views: Vec<vk::ImageView>,
}

impl Swapchain {
    /// create a swapchain sized to `window_size`, negotiating a preferred
    /// surface format, present mode and composite alpha
    pub fn create(
        device: &Device,
        swapchain_fn: &swapchain::Device,
        surface_fn: &ash::khr::surface::Instance,
        physical: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        window_size: PhysicalSize<u32>,
    ) -> Result<Self, Box<dyn Error>> {
        let capabilities =
            unsafe { surface_fn.get_physical_device_surface_capabilities(physical, surface)? };
        let formats = unsafe { surface_fn.get_physical_device_surface_formats(physical, surface)? };
        let present_modes =
            unsafe { surface_fn.get_physical_device_surface_present_modes(physical, surface)? };

        if formats.is_empty() || present_modes.is_empty() {
            return Err("surface supports no formats or present modes".into());
        }

        let format = pick_format(&formats);
        let extent = pick_extent(&capabilities, window_size);

        let image_count = {
            let count = capabilities.min_image_count.saturating_add(1);
            if capabilities.max_image_count > 0 {
                count.min(capabilities.max_image_count)
            } else {
                count
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(pick_composite_alpha(&capabilities))
            // fifo is the best one.
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None)? };
        let images = unsafe { swapchain_fn.get_swapchain_images(swapchain)? };

        let views = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&view_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        log::info!("created swapchain: {}x{}", extent.width, extent.height);
        Ok(Self {
            swapchain,
            format,
            extent,
            views,
        })
    }

    /// destroy the swapchain and its image views
    pub fn destroy(&mut self, device: &Device, swapchain_fn: &swapchain::Device) {
        for &view in &self.views {
            unsafe { device.destroy_image_view(view, None) };
        }
        unsafe { swapchain_fn.destroy_swapchain(self.swapchain, None) };
        log::info!("destroyed swapchain");
    }
}

fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    let preferred = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_UNORM,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
    if formats.contains(&preferred) {
        preferred
    } else {
        formats[0]
    }
}

/// surface extent: use the surface's current extent when fixed, otherwise
/// clamp the window size to the supported range
fn pick_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window_size: PhysicalSize<u32>,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: window_size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn pick_composite_alpha(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::CompositeAlphaFlagsKHR {
    let candidates = [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ];
    candidates
        .into_iter()
        .find(|candidate| capabilities.supported_composite_alpha.contains(*candidate))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE)
}
