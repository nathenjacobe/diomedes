use std::error::Error;
use std::mem::size_of;
use std::slice;

use ash::vk;

use super::buffer::Buffer;
use super::device::DeviceContext;

/// one vertex: position, surface normal (for lighting) and color; the layout
/// is fixed by `vertex::binding_description` and
/// `vertex::attribute_descriptions`, which the pipeline is built from
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let float3 = |location, offset| {
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(location)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset)
        };
        [
            float3(0, 0),
            float3(1, 3 * size_of::<f32>() as u32),
            float3(2, 6 * size_of::<f32>() as u32),
        ]
    }
}

/// vertex data for one mesh in a gpu buffer
pub struct VertexBuffer {
    buffer: Buffer,
}

impl VertexBuffer {
    pub fn create(device: &DeviceContext, vertices: &[Vertex]) -> Result<Self, Box<dyn Error>> {
        let size = (size_of::<Vertex>() * vertices.len()) as vk::DeviceSize;

        let mut buffer = Buffer::create_device_local(
            device,
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let bytes = unsafe { slice::from_raw_parts(vertices.as_ptr() as *const u8, size as usize) };
        buffer.upload(device, bytes)?;

        Ok(Self { buffer })
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        self.buffer.destroy(device);
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.buffer
    }
}

/// index data for one mesh in a gpu buffer
pub struct IndexBuffer {
    buffer: Buffer,
    index_count: u32,
}

impl IndexBuffer {
    pub fn create(device: &DeviceContext, indices: &[u32]) -> Result<Self, Box<dyn Error>> {
        let size = (size_of::<u32>() * indices.len()) as vk::DeviceSize;

        let mut buffer = Buffer::create_device_local(
            device,
            size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let bytes = unsafe { slice::from_raw_parts(indices.as_ptr() as *const u8, size as usize) };
        buffer.upload(device, bytes)?;

        Ok(Self {
            buffer,
            index_count: indices.len() as u32,
        })
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        self.buffer.destroy(device);
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}
