//! host-side gpu physics state and constraint preparation;
//!
//! dynamics integration and constraint solves run in the gpu avbd compute
//! shader; this module retains only body descriptors, gpu contact readback
//! conversion, warmstart state, and the cpu-side container-wall constraints

use std::collections::HashMap;

use glam::{Quat, Vec3};

use super::Shape;
use super::constraints::Constraint;

/// sentinel `b` for contacts against the container box
pub const CONTAINER: usize = usize::MAX;

/// a rigid body descriptor and the state uploaded to the gpu solver
#[derive(Clone, Copy, Debug)]
pub struct AvbdBody {
    pub shape: Shape,
    pub position: Vec3,
    pub orientation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: f32,
    pub inv_moment: Vec3,
    pub friction: f32,
    pub prev_velocity: Vec3,
}

impl AvbdBody {
    pub fn sphere(position: Vec3, radius: f32, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inv_moment = if mass > 0.0 {
            Vec3::splat(1.0 / (0.4 * mass * radius * radius))
        } else {
            Vec3::ZERO
        };
        Self {
            shape: Shape::Sphere(radius),
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_moment,
            friction: 0.8,
            prev_velocity: Vec3::ZERO,
        }
    }

    pub fn cube(position: Vec3, half_extent: f32, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inv_moment = if mass > 0.0 {
            Vec3::splat(1.0 / ((2.0 / 3.0) * mass * half_extent * half_extent))
        } else {
            Vec3::ZERO
        };
        Self {
            shape: Shape::Cube(half_extent),
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_moment,
            friction: 0.8,
            prev_velocity: Vec3::ZERO,
        }
    }

    pub fn tetrahedron(position: Vec3, corners: [Vec3; 4], mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let radius = corners
            .iter()
            .map(|corner| corner.length())
            .fold(0.0, f32::max);
        let inv_moment = if mass > 0.0 {
            Vec3::splat(1.0 / (0.4 * mass * radius * radius))
        } else {
            Vec3::ZERO
        };
        Self {
            shape: Shape::Tetrahedron(corners),
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            inv_mass,
            inv_moment,
            friction: 0.8,
            prev_velocity: Vec3::ZERO,
        }
    }

    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
}

/// a contact record produced from gpu narrow-phase witnesses;
#[derive(Clone, Copy, Debug)]
pub struct AvbdContact {
    pub a: usize,
    pub b: usize,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub r_a: Vec3,
    pub r_b: Vec3,
    pub c0: Vec3,
    pub penalty: Vec3,
    pub lambda: Vec3,
    pub friction: f32,
}

/// a static box whose interior is free space
#[derive(Clone, Copy, Debug)]
pub struct AvbdContainer {
    pub center: Vec3,
    pub rotation: Quat,
    pub half_extent: f32,
}

/// host-side settings uploaded to the gpu solver
#[derive(Clone, Copy, Debug)]
pub struct AvbdOptions {
    pub dt: f32,
    pub gravity: Vec3,
    pub iterations: usize,
    pub alpha: f32,
    pub beta_lin: f32,
    pub margin: f32,
    pub penalty_min: f32,
    pub penalty_max: f32,
    pub gamma: f32,
    pub friction: f32,
}

impl Default for AvbdOptions {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity: Vec3::new(0.0, -10.0, 0.0),
            iterations: 10,
            alpha: 0.99,
            beta_lin: 10000.0,
            margin: 0.01,
            penalty_min: 1.0,
            penalty_max: 1e10,
            gamma: 0.999,
            friction: 0.8,
        }
    }
}

/// gpu simulation state; no cpu integration or constraint solve is exposed
pub struct AvbdSolver {
    pub bodies: Vec<AvbdBody>,
    pub container: Option<AvbdContainer>,
    pub constraints: Vec<Constraint>,
    contacts: Vec<AvbdContact>,
    contact_offsets: Vec<u32>,
    contact_indices: Vec<u32>,
    warm: HashMap<(usize, usize), (Vec3, Vec3)>,
}

impl AvbdSolver {
    pub fn new(bodies: Vec<AvbdBody>) -> Self {
        let count = bodies.len();
        Self {
            bodies,
            container: None,
            constraints: Vec::new(),
            contacts: Vec::new(),
            contact_offsets: Vec::with_capacity(count + 1),
            contact_indices: Vec::new(),
            warm: HashMap::new(),
        }
    }

    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// prepare gpu contact records and warmstart state; the actual solve is
    /// dispatched by [`crate::render::gpu_physics::avbdcompute`]
    pub fn prepare_contacts(&mut self, mut body_contacts: Vec<AvbdContact>, options: &AvbdOptions) {
        if let Some(container) = self.container {
            for (index, body) in self.bodies.iter().enumerate() {
                if !body.is_static() {
                    body_contacts.extend(container_contacts(&container, body, index, options));
                }
            }
        }
        self.finalize_contacts(body_contacts, options);
    }

    pub fn contacts(&self) -> &[AvbdContact] {
        &self.contacts
    }

    pub fn contact_offsets(&self) -> &[u32] {
        &self.contact_offsets
    }

    pub fn contact_indices(&self) -> &[u32] {
        &self.contact_indices
    }

    /// adopt the gpu result and carry contact multipliers into the next step
    pub fn sync_bodies_from_gpu(
        &mut self,
        positions: &[Vec3],
        orientations: &[Quat],
        velocities: &[Vec3],
        angular_velocities: &[Vec3],
        prev_velocities: &[Vec3],
        lambda: &[Vec3],
        penalty: &[Vec3],
    ) {
        for (index, body) in self.bodies.iter_mut().enumerate() {
            if body.is_static() {
                continue;
            }
            body.position = positions[index];
            body.orientation = orientations[index];
            body.velocity = velocities[index];
            body.angular_velocity = angular_velocities[index];
            body.prev_velocity = prev_velocities[index];
        }
        for (contact, (lambda, penalty)) in self.contacts.iter_mut().zip(lambda.iter().zip(penalty))
        {
            contact.lambda = *lambda;
            contact.penalty = *penalty;
        }
    }

    pub fn raw_contact(
        a: usize,
        b: usize,
        body_a: &AvbdBody,
        body_b: &AvbdBody,
        normal: Vec3,
        depth: f32,
        point_a: Vec3,
        point_b: Vec3,
        options: &AvbdOptions,
    ) -> AvbdContact {
        let (tangent1, tangent2) = tangents(normal);
        AvbdContact {
            a,
            b,
            normal: -normal,
            tangent1,
            tangent2,
            r_a: body_a.orientation.inverse() * (point_a - body_a.position),
            r_b: body_b.orientation.inverse() * (point_b - body_b.position),
            c0: Vec3::new(options.margin - depth, 0.0, 0.0),
            penalty: Vec3::splat(options.penalty_min),
            lambda: Vec3::ZERO,
            friction: options.friction,
        }
    }

    fn finalize_contacts(&mut self, contacts: Vec<AvbdContact>, options: &AvbdOptions) {
        self.warm.clear();
        for contact in self.contacts.drain(..) {
            self.warm
                .insert((contact.a, contact.b), (contact.lambda, contact.penalty));
        }
        let mut contacts = contacts;
        for contact in &mut contacts {
            if let Some((lambda, penalty)) = self.warm.remove(&(contact.a, contact.b)) {
                contact.lambda = lambda;
                contact.penalty = penalty;
            }
            contact.lambda *= options.alpha * options.gamma;
            contact.penalty = (contact.penalty * options.gamma).clamp(
                Vec3::splat(options.penalty_min),
                Vec3::splat(options.penalty_max),
            );
        }
        self.contacts = contacts;

        let n = self.bodies.len();
        self.contact_offsets.clear();
        self.contact_offsets.resize(n + 1, 0);
        for contact in &self.contacts {
            self.contact_offsets[contact.a + 1] += 1;
            if contact.b != CONTAINER {
                self.contact_offsets[contact.b + 1] += 1;
            }
        }
        for index in 0..n {
            self.contact_offsets[index + 1] += self.contact_offsets[index];
        }
        self.contact_indices
            .resize(self.contact_offsets[n] as usize, 0);
        let mut cursor = self.contact_offsets.clone();
        for (index, contact) in self.contacts.iter().enumerate() {
            let slot = cursor[contact.a] as usize;
            self.contact_indices[slot] = index as u32;
            cursor[contact.a] += 1;
            if contact.b != CONTAINER {
                let slot = cursor[contact.b] as usize;
                self.contact_indices[slot] = index as u32;
                cursor[contact.b] += 1;
            }
        }
    }
}

fn cube_support_face(body: &AvbdBody, outward: Vec3, half_extent: f32) -> [Vec3; 4] {
    let axes = [
        body.orientation * Vec3::X,
        body.orientation * Vec3::Y,
        body.orientation * Vec3::Z,
    ];
    let mut axis_index = 0usize;
    let mut best_alignment = axes[0].dot(outward).abs();
    for index in 1..3 {
        let alignment = axes[index].dot(outward).abs();
        if alignment > best_alignment {
            axis_index = index;
            best_alignment = alignment;
        }
    }
    let face_axis = axes[axis_index];
    let face_normal = face_axis
        * if face_axis.dot(outward) >= 0.0 {
            1.0
        } else {
            -1.0
        };
    let center = body.position + face_normal * half_extent;
    let (u, v) = if axis_index == 0 {
        (axes[1], axes[2])
    } else if axis_index == 1 {
        (axes[0], axes[2])
    } else {
        (axes[0], axes[1])
    };
    [
        center + u * half_extent + v * half_extent,
        center - u * half_extent + v * half_extent,
        center - u * half_extent - v * half_extent,
        center + u * half_extent - v * half_extent,
    ]
}

fn container_contacts(
    container: &AvbdContainer,
    body: &AvbdBody,
    body_index: usize,
    options: &AvbdOptions,
) -> Vec<AvbdContact> {
    let mut out = Vec::new();
    for (axis, sign) in [
        (Vec3::X, 1.0),
        (Vec3::X, -1.0),
        (Vec3::Y, 1.0),
        (Vec3::Y, -1.0),
        (Vec3::Z, 1.0),
        (Vec3::Z, -1.0),
    ] {
        let outward = container.rotation * (axis * sign);
        let plane = container.center.dot(outward) + container.half_extent;
        let (support_points, support_count) = match body.shape {
            Shape::Cube(half_extent) => (cube_support_face(body, outward, half_extent), 4usize),
            _ => {
                let local_direction = body.orientation.inverse() * outward;
                let support_point =
                    body.orientation * body.shape.support(local_direction) + body.position;
                ([support_point; 4], 1usize)
            }
        };
        let normal = -outward;
        let (tangent1, tangent2) = tangents(normal);
        for point_index in 0..support_count {
            let contact_point = support_points[point_index];
            let current_depth = contact_point.dot(outward) - plane;
            let predicted_point = contact_point + body.velocity * options.dt;
            let predicted_depth = predicted_point.dot(outward) - plane;
            let depth = current_depth.max(predicted_depth);
            if depth > 0.0 {
                out.push(AvbdContact {
                    a: body_index,
                    b: CONTAINER,
                    normal,
                    tangent1,
                    tangent2,
                    r_a: body.orientation.inverse() * (contact_point - body.position),
                    r_b: Vec3::ZERO,
                    c0: Vec3::new(options.margin - depth, 0.0, 0.0),
                    penalty: Vec3::splat(options.penalty_min),
                    lambda: Vec3::ZERO,
                    friction: options.friction,
                });
            }
        }
    }
    out
}

fn tangents(normal: Vec3) -> (Vec3, Vec3) {
    let tangent1 = if normal.x.abs() > 0.9 {
        normal.cross(Vec3::Y)
    } else {
        normal.cross(Vec3::X)
    }
    .normalize();
    (tangent1, normal.cross(tangent1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penetrating_cube_gets_four_floor_contacts() {
        let container = AvbdContainer {
            center: Vec3::new(0.0, 4.0, 0.0),
            rotation: Quat::IDENTITY,
            half_extent: 4.0,
        };
        let body = AvbdBody::cube(Vec3::new(0.0, 0.49, 0.0), 0.5, 1.0);
        let options = AvbdOptions::default();
        let contacts = container_contacts(&container, &body, 0, &options);
        assert_eq!(contacts.len(), 4);
        assert!(contacts.iter().all(|contact| contact.normal == Vec3::Y));
        assert!(contacts.iter().all(|contact| contact.r_a.y < -0.49));
    }
}
