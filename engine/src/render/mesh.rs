use std::error::Error;

use crate::asset;
use crate::render::vertex::Vertex;
use crate::scene::MeshShape;

impl MeshShape {
    /// generate the geometry for this shape
    pub(crate) fn generated_geometry(&self) -> Result<asset::Mesh, Box<dyn Error>> {
        match self {
            MeshShape::Triangle => Ok(triangle()),
            MeshShape::Cube => Ok(cube()),
            MeshShape::Icosphere => Err(
                "no geometry registered for MeshShape::Icosphere; call Renderer::register_mesh_data first"
                    .into(),
            ),
            MeshShape::Tetrahedron => Err(
                "no geometry registered for MeshShape::Tetrahedron; call Renderer::register_mesh_data first"
                    .into(),
            ),
        }
    }
}

/// three vertices in ndc, one primary color per corner
fn triangle() -> asset::Mesh {
    asset::Mesh {
        vertices: vec![
            Vertex {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [0.0, 0.0, 1.0],
            },
        ],
        indices: vec![0, 1, 2],
    }
}

/// a unit cube (half-extent 0;5) with 24 vertices; four per face so each
/// face has a uniform color, and 36 indices
fn cube() -> asset::Mesh {
    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
    const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
    const MAGENTA: [f32; 3] = [1.0, 0.0, 1.0];
    const CYAN: [f32; 3] = [0.0, 1.0, 1.0];

    let face =
        |corner: [glam::Vec3; 4], normal: [f32; 3], color: [f32; 3]| -> ([Vertex; 4], [u32; 6]) {
            let vertices = corner.map(|position| Vertex {
                position: position.to_array(),
                normal,
                color,
            });
            (vertices, [0, 1, 2, 0, 2, 3])
        };

    // one quad per axis direction, wound counter-clockwise when viewed from
    // outside; the pipeline culls back faces with front_face = ccw
    let (px, pi) = face(
        [
            glam::Vec3::new(0.5, -0.5, -0.5),
            glam::Vec3::new(0.5, 0.5, -0.5),
            glam::Vec3::new(0.5, 0.5, 0.5),
            glam::Vec3::new(0.5, -0.5, 0.5),
        ],
        [1.0, 0.0, 0.0],
        RED,
    );
    let (nx, ni) = face(
        [
            glam::Vec3::new(-0.5, -0.5, 0.5),
            glam::Vec3::new(-0.5, 0.5, 0.5),
            glam::Vec3::new(-0.5, 0.5, -0.5),
            glam::Vec3::new(-0.5, -0.5, -0.5),
        ],
        [-1.0, 0.0, 0.0],
        CYAN,
    );
    let (py, pyi) = face(
        [
            glam::Vec3::new(-0.5, 0.5, -0.5),
            glam::Vec3::new(-0.5, 0.5, 0.5),
            glam::Vec3::new(0.5, 0.5, 0.5),
            glam::Vec3::new(0.5, 0.5, -0.5),
        ],
        [0.0, 1.0, 0.0],
        GREEN,
    );
    let (ny, nyi) = face(
        [
            glam::Vec3::new(-0.5, -0.5, 0.5),
            glam::Vec3::new(-0.5, -0.5, -0.5),
            glam::Vec3::new(0.5, -0.5, -0.5),
            glam::Vec3::new(0.5, -0.5, 0.5),
        ],
        [0.0, -1.0, 0.0],
        MAGENTA,
    );
    let (pz, pzi) = face(
        [
            glam::Vec3::new(-0.5, -0.5, 0.5),
            glam::Vec3::new(0.5, -0.5, 0.5),
            glam::Vec3::new(0.5, 0.5, 0.5),
            glam::Vec3::new(-0.5, 0.5, 0.5),
        ],
        [0.0, 0.0, 1.0],
        BLUE,
    );
    let (nz, nzi) = face(
        [
            glam::Vec3::new(0.5, -0.5, -0.5),
            glam::Vec3::new(-0.5, -0.5, -0.5),
            glam::Vec3::new(-0.5, 0.5, -0.5),
            glam::Vec3::new(0.5, 0.5, -0.5),
        ],
        [0.0, 0.0, -1.0],
        YELLOW,
    );

    let faces = [px, nx, py, ny, pz, nz];
    let indices = [pi, ni, pyi, nyi, pzi, nzi];

    let mut vertices = Vec::with_capacity(24);
    let mut all_indices = Vec::with_capacity(36);
    for (index, (face_vertices, face_indices)) in faces.into_iter().zip(indices).enumerate() {
        let base = (index * 4) as u32;
        vertices.extend_from_slice(&face_vertices);
        all_indices.extend(face_indices.iter().map(|i| base + i));
    }

    asset::Mesh {
        vertices,
        indices: all_indices,
    }
}
