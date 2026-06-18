use std::error::Error;
use std::mem::size_of;
use std::slice;

use ash::vk;
use glam::Mat4;

use super::buffer::Buffer;
use super::device::DeviceContext;

/// host-visible uniform buffer has one `mat4` mvp per scene instance,
/// indexed by the shader through `gl_instanceindex`; the capacity is fixed at
/// creation (see `max_instances` in context;rs); scenes must stay within it
pub struct UniformBuffer {
    buffer: Buffer,
    pub capacity: usize,
}

impl UniformBuffer {
    pub fn create(device: &DeviceContext, capacity: usize) -> Result<Self, Box<dyn Error>> {
        // each instance slot holds two matrices (mvp + model)
        // c.f. the shader's interleaved `transforms[]` array
        let size = (capacity * 2 * size_of::<Mat4>()) as vk::DeviceSize;
        let buffer = Buffer::create(
            device,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        Ok(Self { buffer, capacity })
    }

    /// overwrite the buffer contents with the given matrices, packed
    pub fn write(&mut self, device: &DeviceContext, data: &[Mat4]) -> Result<(), Box<dyn Error>> {
        if data.is_empty() {
            return Ok(());
        }
        if data.len() > self.capacity * 2 {
            return Err("too many instances for the uniform buffer".into());
        }
        let size = size_of::<Mat4>() * data.len();
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, size) };
        self.buffer.write(device, bytes)
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        self.buffer.destroy(device);
    }
}
