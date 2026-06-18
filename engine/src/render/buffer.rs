use std::error::Error;
use std::ffi::c_void;

use ash::vk;

use super::device::DeviceContext;

/// a vulkan buffer with its backing device memory
///
/// memory is allocated from the type that satisfies `properties`; host-visible
/// memory can be written directly with `buffer::write`
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl Buffer {
    pub fn create(
        device: &DeviceContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self, Box<dyn Error>> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.device.create_buffer(&create_info, None)? };

        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let memory_type = device.find_memory_type(requirements, properties)?;

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);

        let memory = unsafe { device.device.allocate_memory(&allocate_info, None)? };
        unsafe { device.device.bind_buffer_memory(buffer, memory, 0)? };

        Ok(Self { buffer, memory })
    }

    /// create a device-local buffer (fastest access for the gpu; uploaded
    /// through a host-visible staging buffer);
    pub fn create_device_local(
        device: &DeviceContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, Box<dyn Error>> {
        Self::create(device, size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    /// copy raw bytes into the buffer (requires host-visible memory);
    pub fn write(&mut self, device: &DeviceContext, data: &[u8]) -> Result<(), Box<dyn Error>> {
        if data.is_empty() {
            return Ok(());
        }
        let size = data.len() as u64;
        let ptr = unsafe {
            device
                .device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())?
        };
        unsafe { ptr.copy_from_nonoverlapping(data.as_ptr() as *const c_void, data.len()) };
        unsafe { device.device.unmap_memory(self.memory) };
        Ok(())
    }

    /// copy `size` raw bytes out of the buffer
    pub fn read(&self, device: &DeviceContext, size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
        // vkmapmemory rejects zero-length ranges; an empty read is thus a no-op
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut data = vec![0u8; size];
        let ptr = unsafe {
            device.device.map_memory(
                self.memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )?
        };
        unsafe { ptr.copy_to_nonoverlapping(data.as_mut_ptr() as *mut c_void, size) };
        unsafe { device.device.unmap_memory(self.memory) };
        Ok(data)
    }

    /// copy `data` into the buffer through a host-visible staging buffer
    /// the buffer must have been created with `transfer_dst` usage
    pub fn upload(&mut self, device: &DeviceContext, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let size = data.len() as vk::DeviceSize;

        let mut staging = Buffer::create(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging.write(device, data)?;

        // one-shot copy on the graphics queue (the only queue we create lol)
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.device.create_command_pool(&pool_info, None)? };

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.device.allocate_command_buffers(&allocate_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        let copies = [copy];
        unsafe {
            device
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;
            device
                .device
                .cmd_copy_buffer(command_buffer, staging.buffer, self.buffer, &copies);
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
        log::info!("uploaded {} bytes to device-local buffer", data.len());
        Ok(())
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        unsafe {
            device.device.destroy_buffer(self.buffer, None);
            device.device.free_memory(self.memory, None);
        }
        log::info!("destroyed buffer");
    }
}
