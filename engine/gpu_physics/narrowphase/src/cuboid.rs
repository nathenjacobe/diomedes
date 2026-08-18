use spirv_std::glam::Vec3;

use crate::body::GpuBody;
use crate::support::CUBE;

const AXIS_EPSILON: f32 = 1.0e-8;
const CLIP_EPSILON: f32 = 1.0e-5;
const MAX_CONTACTS: usize = 4;

/// up to four face contacts for one cube pair
pub struct Manifold {
    pub normal: Vec3,
    pub depth: f32,
    pub point_a: [Vec3; MAX_CONTACTS],
    pub point_b: [Vec3; MAX_CONTACTS],
    pub count: u32,
}

fn test_axis(
    raw_axis: Vec3,
    kind: u32,
    index: u32,
    a_axes: &[Vec3; 3],
    b_axes: &[Vec3; 3],
    half_a: f32,
    half_b: f32,
    delta: Vec3,
    best_overlap: &mut f32,
    best_normal: &mut Vec3,
    best_kind: &mut u32,
    best_index: &mut u32,
) -> bool {
    if raw_axis.length_squared() <= AXIS_EPSILON {
        return true;
    }
    let axis = raw_axis.normalize();
    let radius_a = half_a
        * (axis.dot(a_axes[0]).abs() + axis.dot(a_axes[1]).abs() + axis.dot(a_axes[2]).abs());
    let radius_b = half_b
        * (axis.dot(b_axes[0]).abs() + axis.dot(b_axes[1]).abs() + axis.dot(b_axes[2]).abs());
    let overlap = radius_a + radius_b - delta.dot(axis).abs();
    if overlap < 0.0 {
        return false;
    }
    if overlap < *best_overlap {
        *best_overlap = overlap;
        *best_normal = if delta.dot(axis) >= 0.0 { axis } else { -axis };
        *best_kind = kind;
        *best_index = index;
    }
    true
}

fn sat(a: &GpuBody, b: &GpuBody) -> Option<(Vec3, f32, u32, u32)> {
    let a_axes = [a.quat() * Vec3::X, a.quat() * Vec3::Y, a.quat() * Vec3::Z];
    let b_axes = [b.quat() * Vec3::X, b.quat() * Vec3::Y, b.quat() * Vec3::Z];
    let delta = b.pos() - a.pos();
    let half_a = a.shape.corners[0].x;
    let half_b = b.shape.corners[0].x;
    let mut best_overlap = f32::MAX;
    let mut best_normal = Vec3::X;
    let mut best_kind = 0;
    let mut best_index = 0;

    if !test_axis(
        a_axes[0],
        0,
        0,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) || !test_axis(
        a_axes[1],
        0,
        1,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) || !test_axis(
        a_axes[2],
        0,
        2,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) || !test_axis(
        b_axes[0],
        1,
        0,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) || !test_axis(
        b_axes[1],
        1,
        1,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) || !test_axis(
        b_axes[2],
        1,
        2,
        &a_axes,
        &b_axes,
        half_a,
        half_b,
        delta,
        &mut best_overlap,
        &mut best_normal,
        &mut best_kind,
        &mut best_index,
    ) {
        return None;
    }
    for i in 0..3 {
        for j in 0..3 {
            if !test_axis(
                a_axes[i].cross(b_axes[j]),
                2,
                (i * 3 + j) as u32,
                &a_axes,
                &b_axes,
                half_a,
                half_b,
                delta,
                &mut best_overlap,
                &mut best_normal,
                &mut best_kind,
                &mut best_index,
            ) {
                return None;
            }
        }
    }
    Some((best_normal, best_overlap, best_kind, best_index))
}

fn face_frame(
    body: &GpuBody,
    axis_index: u32,
    outward: Vec3,
    vertices: &mut [Vec3; 4],
) -> (Vec3, Vec3, Vec3, f32, f32) {
    let axes = [
        body.quat() * Vec3::X,
        body.quat() * Vec3::Y,
        body.quat() * Vec3::Z,
    ];
    let half = body.shape.corners[0].x;
    let axis = axes[axis_index as usize];
    let face_normal = axis * if axis.dot(outward) >= 0.0 { 1.0 } else { -1.0 };
    let center = body.pos() + face_normal * half;
    let (u, v) = if axis_index == 0 {
        (axes[1], axes[2])
    } else if axis_index == 1 {
        (axes[0], axes[2])
    } else {
        (axes[0], axes[1])
    };
    vertices[0] = center + u * half + v * half;
    vertices[1] = center - u * half + v * half;
    vertices[2] = center - u * half - v * half;
    vertices[3] = center + u * half - v * half;
    (center, u, v, half, half)
}

fn incident_axis(body: &GpuBody, reference_normal: Vec3) -> u32 {
    let axes = [
        body.quat() * Vec3::X,
        body.quat() * Vec3::Y,
        body.quat() * Vec3::Z,
    ];
    let mut best_axis = 0;
    let mut best_dot = axes[0].dot(reference_normal).abs();
    for i in 1..3 {
        let dot = axes[i].dot(reference_normal).abs();
        if dot > best_dot {
            best_dot = dot;
            best_axis = i;
        }
    }
    best_axis as u32
}

fn clip_plane(
    input: &[Vec3; 8],
    input_count: u32,
    normal: Vec3,
    offset: f32,
    output: &mut [Vec3; 8],
) -> u32 {
    if input_count == 0 {
        return 0;
    }
    let mut count = 0u32;
    let mut previous = input[(input_count - 1) as usize];
    let mut previous_distance = normal.dot(previous) - offset;
    for i in 0..input_count {
        let current = input[i as usize];
        let current_distance = normal.dot(current) - offset;
        let previous_inside = previous_distance <= CLIP_EPSILON;
        let current_inside = current_distance <= CLIP_EPSILON;
        if previous_inside != current_inside {
            let denominator = previous_distance - current_distance;
            if denominator.abs() > AXIS_EPSILON && count < 8 {
                let t = (previous_distance / denominator).clamp(0.0, 1.0);
                output[count as usize] = previous + (current - previous) * t;
                count += 1;
            }
        }
        if current_inside && count < 8 {
            output[count as usize] = current;
            count += 1;
        }
        previous = current;
        previous_distance = current_distance;
    }
    count
}

fn centered_contact(a: &GpuBody, b: &GpuBody, normal: Vec3, depth: f32) -> Manifold {
    let a_axes = [a.quat() * Vec3::X, a.quat() * Vec3::Y, a.quat() * Vec3::Z];
    let b_axes = [b.quat() * Vec3::X, b.quat() * Vec3::Y, b.quat() * Vec3::Z];
    let half_a = a.shape.corners[0].x;
    let half_b = b.shape.corners[0].x;
    let radius_a = half_a
        * (normal.dot(a_axes[0]).abs() + normal.dot(a_axes[1]).abs() + normal.dot(a_axes[2]).abs());
    let radius_b = half_b
        * (normal.dot(b_axes[0]).abs() + normal.dot(b_axes[1]).abs() + normal.dot(b_axes[2]).abs());
    let mut point_a = [Vec3::ZERO; MAX_CONTACTS];
    let mut point_b = [Vec3::ZERO; MAX_CONTACTS];
    point_a[0] = a.pos() + normal * radius_a;
    point_b[0] = b.pos() - normal * radius_b;
    Manifold {
        normal,
        depth,
        point_a,
        point_b,
        count: 1,
    }
}

/// compute a stable cube manifold, falling back to one centered contact for
/// edge-edge configurations
pub fn collide_manifold(a: &GpuBody, b: &GpuBody) -> Option<Manifold> {
    if a.shape.tag != CUBE || b.shape.tag != CUBE {
        return None;
    }
    let (normal, depth, kind, index) = sat(a, b)?;
    if kind == 2 {
        return Some(centered_contact(a, b, normal, depth));
    }

    let reference_is_a = kind == 0;
    let reference_normal = if reference_is_a { normal } else { -normal };
    let reference_body = if reference_is_a { *a } else { *b };
    let incident_body = if reference_is_a { *b } else { *a };
    let reference_axis = index;
    let incident_axis = incident_axis(&incident_body, reference_normal);

    let mut reference_face = [Vec3::ZERO; 4];
    let (reference_center, u, v, extent_u, extent_v) = face_frame(
        &reference_body,
        reference_axis,
        reference_normal,
        &mut reference_face,
    );
    let mut polygon_a = [Vec3::ZERO; 8];
    let mut polygon_b = [Vec3::ZERO; 8];
    let mut incident_face = [Vec3::ZERO; 4];
    face_frame(
        &incident_body,
        incident_axis,
        -reference_normal,
        &mut incident_face,
    );
    for i in 0..4 {
        polygon_a[i] = incident_face[i];
    }
    let mut count = 4u32;
    count = clip_plane(
        &polygon_a,
        count,
        u,
        u.dot(reference_center) + extent_u,
        &mut polygon_b,
    );
    count = clip_plane(
        &polygon_b,
        count,
        -u,
        (-u).dot(reference_center) + extent_u,
        &mut polygon_a,
    );
    count = clip_plane(
        &polygon_a,
        count,
        v,
        v.dot(reference_center) + extent_v,
        &mut polygon_b,
    );
    count = clip_plane(
        &polygon_b,
        count,
        -v,
        (-v).dot(reference_center) + extent_v,
        &mut polygon_a,
    );
    if count == 0 {
        return Some(centered_contact(a, b, normal, depth));
    }

    let reference_plane = reference_normal.dot(reference_center);
    let mut point_a = [Vec3::ZERO; MAX_CONTACTS];
    let mut point_b = [Vec3::ZERO; MAX_CONTACTS];
    let mut contact_count = 0u32;
    for i in 0..count {
        if contact_count >= MAX_CONTACTS as u32 {
            break;
        }
        let incident_point = polygon_a[i as usize];
        let point_depth = reference_plane - reference_normal.dot(incident_point);
        if point_depth < -CLIP_EPSILON {
            continue;
        }
        let reference_point = incident_point + reference_normal * point_depth.max(0.0);
        if reference_is_a {
            point_a[contact_count as usize] = reference_point;
            point_b[contact_count as usize] = incident_point;
        } else {
            point_a[contact_count as usize] = incident_point;
            point_b[contact_count as usize] = reference_point;
        }
        contact_count += 1;
    }
    if contact_count == 0 {
        return Some(centered_contact(a, b, normal, depth));
    }
    Some(Manifold {
        normal,
        depth,
        point_a,
        point_b,
        count: contact_count,
    })
}

#[cfg(test)]
mod tests {
    use spirv_std::glam::{Quat, Vec4};

    use super::*;
    use crate::support::ShapeData;

    fn cube(position: Vec3) -> GpuBody {
        GpuBody {
            position: position.extend(0.0),
            orientation: Quat::IDENTITY.to_array().into(),
            shape: ShapeData {
                tag: CUBE,
                _pad: [0; 3],
                corners: [
                    Vec4::new(0.5, 0.5, 0.5, 0.0),
                    Vec4::ZERO,
                    Vec4::ZERO,
                    Vec4::ZERO,
                ],
            },
        }
    }

    #[test]
    fn touching_cubes_have_zero_penetration() {
        let manifold = collide_manifold(&cube(Vec3::ZERO), &cube(Vec3::X)).unwrap();
        assert!(manifold.depth.abs() < 1.0e-6);
        assert!(manifold.normal.dot(Vec3::X) > 0.99);
    }

    #[test]
    fn overlapping_cubes_report_four_face_contacts() {
        let manifold =
            collide_manifold(&cube(Vec3::ZERO), &cube(Vec3::new(0.9, 0.0, 0.0))).unwrap();
        assert!((manifold.depth - 0.1).abs() < 1.0e-6);
        assert!(manifold.normal.dot(Vec3::X) > 0.99);
        assert_eq!(manifold.count, 4);
    }

    #[test]
    fn separated_cubes_have_no_contact() {
        assert!(collide_manifold(&cube(Vec3::ZERO), &cube(Vec3::new(1.1, 0.0, 0.0))).is_none());
    }
}
