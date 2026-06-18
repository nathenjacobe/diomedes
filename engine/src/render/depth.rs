use std::error::Error;

use ash::vk;

use super::device::DeviceContext;

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// depth attachment for the swapchain-sized framebuffer; recreated with the
/// swapchain (via [`frame`](super::frame::frame)) on resize
pub struct DepthBuffer {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl DepthBuffer {
    pub fn create(device: &DeviceContext, extent: vk::Extent2D) -> Result<Self, Box<dyn Error>> {
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_FORMAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.device.create_image(&create_info, None)? };

        let requirements = unsafe { device.device.get_image_memory_requirements(image) };
        let memory_type =
            device.find_memory_type(requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { device.device.allocate_memory(&allocate_info, None)? };
        unsafe { device.device.bind_image_memory(image, memory, 0)? };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(DEPTH_FORMAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.device.create_image_view(&view_info, None)? };

        Ok(Self {
            image,
            view,
            memory,
        })
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
            device.device.free_memory(self.memory, None);
        }
        log::info!("destroyed depth buffer");
    }

    pub fn handle(&self) -> vk::ImageView {
        self.view
    }
}
