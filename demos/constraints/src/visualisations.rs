use diomedes::glam::{Quat, Vec3};
use diomedes::scene::{InstanceId, MeshShape, Scene, Transform};

/// a constraint link rendered as short cube segments
pub struct ConstraintVisualisation {
    segments: Vec<InstanceId>,
    amplitude: f32,
    thickness: f32,
}

impl ConstraintVisualisation {
    pub fn new(scene: &mut Scene, segment_count: usize, amplitude: f32) -> Self {
        let mut segments = Vec::with_capacity(segment_count.max(2));
        for _ in 0..segment_count.max(2) {
            segments.push(scene.add_shape(
                MeshShape::Cube,
                Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.05)),
            ));
        }
        Self {
            segments,
            amplitude,
            thickness: 0.08,
        }
    }

    pub fn update(&self, scene: &mut Scene, start: Vec3, end: Vec3) {
        let delta = end - start;
        let direction = delta.normalize_or_zero();
        let reference = if direction.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let side = direction.cross(reference).normalize_or_zero();
        let count = self.segments.len();

        for (index, instance_id) in self.segments.iter().copied().enumerate() {
            let t0 = index as f32 / count as f32;
            let t1 = (index + 1) as f32 / count as f32;
            let zigzag = |t: f32, point_index: usize| {
                let offset = if point_index == 0 || point_index == count {
                    0.0
                } else if point_index % 2 == 0 {
                    self.amplitude
                } else {
                    -self.amplitude
                };
                start + delta * t + side * offset
            };
            let a = zigzag(t0, index);
            let b = zigzag(t1, index + 1);
            let segment = b - a;
            let length = segment.length().max(0.001);
            let rotation = if segment.length_squared() > 1.0e-8 {
                Quat::from_rotation_arc(Vec3::X, segment / length)
            } else {
                Quat::IDENTITY
            };
            if let Some(instance) = scene.instance_mut(instance_id) {
                instance.transform = Transform::new(
                    (a + b) * 0.5,
                    rotation,
                    Vec3::new(length, self.thickness, self.thickness),
                );
            }
        }
    }
}
