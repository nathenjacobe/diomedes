use std::collections::HashMap;
use std::error::Error;

use ash::vk;

use crate::asset;
use crate::render::device::DeviceContext;
use crate::render::vertex::{IndexBuffer, VertexBuffer};
use crate::scene::MeshShape;

/// interned mesh geometry; each distinct shape is uploaded to the gpu once
/// and shared by every instance of that shape; this is what makes batched /
/// instanced draws cheap; geometry is never duplicated per instance
/// I believe this is called "kitbashing" in the industry
pub struct MeshLibrary {
    meshes: HashMap<MeshShape, InternedMesh>,
}

pub struct InternedMesh {
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
}

impl MeshLibrary {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
        }
    }

    /// upload geometry for `shape` into the gpu, unless already interned;
    pub fn intern(
        &mut self,
        device: &DeviceContext,
        shape: MeshShape,
        mesh: &asset::Mesh,
    ) -> Result<(), Box<dyn Error>> {
        if self.meshes.contains_key(&shape) {
            return Ok(());
        }
        let vertex_buffer = VertexBuffer::create(device, &mesh.vertices)?;
        let index_buffer = IndexBuffer::create(device, &mesh.indices)?;
        log::info!("interned mesh {shape:?}");
        self.meshes.insert(
            shape,
            InternedMesh {
                vertex_buffer,
                index_buffer,
            },
        );
        Ok(())
    }

    /// return the interned geometry for `shape`, generating built-in shapes
    /// on first use; `icosphere` must be registered first via `self::intern`
    pub fn get_or_intern(
        &mut self,
        device: &DeviceContext,
        shape: MeshShape,
    ) -> Result<&InternedMesh, Box<dyn Error>> {
        if !self.meshes.contains_key(&shape) {
            let data = shape.generated_geometry()?;
            self.intern(device, shape.clone(), &data)?;
        }
        Ok(self.meshes.get(&shape).unwrap())
    }

    pub fn destroy(&mut self, device: &DeviceContext) {
        for interned in self.meshes.values_mut() {
            interned.vertex_buffer.destroy(device);
            interned.index_buffer.destroy(device);
        }
        self.meshes.clear();
        log::info!("destroyed mesh library");
    }
}

impl InternedMesh {
    pub fn vertex_buffer(&self) -> vk::Buffer {
        self.vertex_buffer.handle()
    }

    pub fn index_buffer(&self) -> vk::Buffer {
        self.index_buffer.handle()
    }

    pub fn index_count(&self) -> u32 {
        self.index_buffer.index_count()
    }
}
