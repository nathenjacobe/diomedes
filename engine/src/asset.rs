//! asset loading w tobj
//! cpu-side mesh data independent of any graphics resources

use std::error::Error;
use std::path::Path;

use crate::render::vertex::Vertex;

/// cpu-side mesh geometry: interleaved vertices and triangle/line indices
/// no gpu resources; intern it into the renderer when ready
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// load an obj file into a single interleaved `mesh`; triangles are
/// triangulated (lol), and all index streams are merged into one vertex/index
/// pair (tobj's `single_index` mode)
/// .obj has no vertex colors, so every vertex gets the given default color;
pub fn load_obj(path: impl AsRef<Path>, color: [f32; 3]) -> Result<Mesh, Box<dyn Error>> {
    let options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ignore_points: true,
        ignore_lines: true,
        ..Default::default()
    };
    let (models, _materials) = tobj::load_obj(path.as_ref(), &options)?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for model in &models {
        let mesh = &model.mesh;
        let vertex_count = mesh.positions.len() / 3;
        vertices.extend((0..vertex_count).map(|i| {
            // obj normals align with the merged vertices in single_index
            // mode when present; otherwise derive one from the position
            let normal = if !mesh.normals.is_empty() {
                [
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]
            } else {
                glam::Vec3::new(
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                )
                .normalize_or_zero()
                .to_array()
            };
            Vertex {
                position: [
                    mesh.positions[i * 3],
                    mesh.positions[i * 3 + 1],
                    mesh.positions[i * 3 + 2],
                ],
                normal,
                color,
            }
        }));
        indices.extend_from_slice(&mesh.indices);
    }

    if vertices.is_empty() {
        return Err("OBJ contains no vertices".into());
    }
    Ok(Mesh { vertices, indices })
}
