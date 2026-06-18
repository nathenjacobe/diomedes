//! convex shapes with support functions used by gpu collision kernels

use glam::Vec3;

/// a convex shape encoded for the gpu solver
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// solid sphere of the given radius
    Sphere(f32),
    /// axis-aligned cube with the given half-extent (sides of length
    /// `2 * half_extent`)
    Cube(f32),
    /// tetrahedron with the given local-space corners
    Tetrahedron([Vec3; 4]),
}

impl Shape {
    /// return the farthest local-space point along `direction`; this is the
    /// only shape query needed by the gpu collision kernels
    pub fn support(&self, direction: Vec3) -> Vec3 {
        match self {
            Shape::Sphere(radius) => direction.normalize_or_zero() * *radius,
            Shape::Cube(half_extent) => Vec3::new(
                if direction.x >= 0.0 {
                    *half_extent
                } else {
                    -*half_extent
                },
                if direction.y >= 0.0 {
                    *half_extent
                } else {
                    -*half_extent
                },
                if direction.z >= 0.0 {
                    *half_extent
                } else {
                    -*half_extent
                },
            ),
            Shape::Tetrahedron(corners) => corners
                .iter()
                .max_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
                .copied()
                .unwrap_or(Vec3::ZERO),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_support_tracks_direction() {
        let shape = Shape::Sphere(2.0);
        assert_eq!(
            shape.support(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(2.0, 0.0, 0.0)
        );
        assert_eq!(
            shape.support(Vec3::new(0.0, -3.0, 0.0)),
            Vec3::new(0.0, -2.0, 0.0)
        );
    }

    #[test]
    fn cube_support_is_a_corner() {
        let shape = Shape::Cube(0.5);
        assert_eq!(shape.support(Vec3::new(1.0, 1.0, 1.0)), Vec3::splat(0.5));
        assert_eq!(
            shape.support(Vec3::new(-1.0, 1.0, 0.0)),
            Vec3::new(-0.5, 0.5, 0.5)
        );
    }

    #[test]
    fn tetrahedron_support_is_the_farthest_corner() {
        let corners = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let shape = Shape::Tetrahedron(corners);
        // several corners tie along +x (dot = 1); any of them is a valid
        // support point, so check the distance, not the exact corner
        let point = shape.support(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(point.x, 1.0);
        assert_eq!(point.dot(Vec3::new(1.0, 0.0, 0.0)), 1.0);
        assert_eq!(
            shape.support(Vec3::new(-1.0, -1.0, 0.0)),
            Vec3::new(-1.0, -1.0, 1.0)
        );
    }
}
