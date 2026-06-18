use std::error::Error;

use ash::vk;

use crate::render::buffer::Buffer;
use crate::render::device::DeviceContext;

/// the font atlas as a sampled gpu image
pub(crate) struct AtlasTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

impl AtlasTexture {
    pub fn create(device: &DeviceContext, width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
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
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
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
    }
}

/// upload pixel data into an image region with a one-shot transfer
/// undefined --> transfer_dst --> copy --> shader_read
pub(crate) fn upload_image(
    device: &DeviceContext,
    image: vk::Image,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut staging = Buffer::create(
        device,
        pixels.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.write(device, pixels)?;

    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(device.queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { device.device.create_command_pool(&pool_info, None)? };
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.device.allocate_command_buffers(&allocate_info)? }[0];

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let to_transfer = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let to_transfers = [to_transfer];
    let to_read = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range);
    let to_reads = [to_read];

    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: 0,
        },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    let regions = [region];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .device
            .begin_command_buffer(command_buffer, &begin_info)?;
        device.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_transfers,
        );
        device.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        device.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_reads,
        );
        device.device.end_command_buffer(command_buffer)?;
    }

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { device.device.create_fence(&fence_info, None)? };
    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
    unsafe {
        device.device.queue_submit(device.queue, &[submit], fence)?;
        device.device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.device.destroy_fence(fence, None);
        device.device.destroy_command_pool(pool, None);
    }
    staging.destroy(device);
    Ok(())
}
