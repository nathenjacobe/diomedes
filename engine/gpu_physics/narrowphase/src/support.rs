//! shape encoding and support functions for the gpu narrow phase

use spirv_std::glam::{Vec3, Vec4};

pub const SPHERE: u32 = 0;
pub const CUBE: u32 = 1;
pub const TETRAHEDRON: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
// TODO: this is weird and not really scalable. need to think of a better solution!
pub struct ShapeData {
    pub tag: u32,
    pub _pad: [u32; 3],
    /// sphere: radius in `.x`; cube: half-extent in `.x`;
    /// tetrahedron: the four corners (as `vec4`, w unused);
    pub corners: [Vec4; 4],
}

/// farthest local-space point of the shape along `direction`
pub fn support(shape: &ShapeData, direction: Vec3) -> Vec3 {
    match shape.tag {
        SPHERE => direction.normalize_or_zero() * shape.corners[0].x,
        CUBE => Vec3::new(
            if direction.x >= 0.0 {
                shape.corners[0].x
            } else {
                -shape.corners[0].x
            },
            if direction.y >= 0.0 {
                shape.corners[0].y
            } else {
                -shape.corners[0].y
            },
            if direction.z >= 0.0 {
                shape.corners[0].z
            } else {
                -shape.corners[0].z
            },
        ),
        TETRAHEDRON => {
            let mut best = shape.corners[0].truncate();
            let mut best_dot = best.dot(direction);
            for i in 1..4 {
                let corner = shape.corners[i as usize].truncate();
                let dot = corner.dot(direction);
                // the cpu's `max_by(total_cmp)` keeps the first of equally
                // maximal corners; matching it exactly matters because the
                // regular tetrahedron produces exact ties on symmetric
                // directions, and a different corner diverges the whole
                // simplex and epa
                if dot > best_dot {
                    best_dot = dot;
                    best = corner;
                }
            }
            best
        }
        _ => Vec3::ZERO,
    }
}

/// whether two shapes are both spheres (the analytic fast path)
pub fn both_spheres(a: &ShapeData, b: &ShapeData) -> bool {
    a.tag == SPHERE && b.tag == SPHERE
}
