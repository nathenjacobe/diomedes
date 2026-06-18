use glam::Vec3;

use super::AvbdBody;

pub const DEFAULT_ROD_STIFFNESS: f32 = 1_000.0;
pub const DEFAULT_SOCKET_STIFFNESS: f32 = 1_000.0;
pub const DEFAULT_ROPE_STIFFNESS: f32 = 1_000.0;

#[derive(Clone, Copy, Debug)]
pub struct Spring {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub rest_length: f32,
    pub stiffness: f32,
}

impl Spring {
    pub fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3,
        anchor_b: Vec3,
        stiffness: f32,
        rest_length: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            stiffness,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rod {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub rest_length: f32,
    pub stiffness: f32,
}

impl Rod {
    pub fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            rest_length,
            stiffness: DEFAULT_ROD_STIFFNESS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BallSocket {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub stiffness: f32,
}

impl BallSocket {
    pub fn new(body_a: usize, body_b: usize, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            stiffness: DEFAULT_SOCKET_STIFFNESS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub body_a: usize,
    pub body_b: usize,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub max_length: f32,
    pub stiffness: f32,
}

impl Rope {
    pub fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3,
        anchor_b: Vec3,
        max_length: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            max_length,
            stiffness: DEFAULT_ROPE_STIFFNESS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    Spring(Spring),
    Rod(Rod),
    BallSocket(BallSocket),
    Rope(Rope),
}

impl Constraint {
    pub fn endpoints(&self, bodies: &[AvbdBody]) -> Option<(Vec3, Vec3)> {
        let (body_a, body_b, anchor_a, anchor_b) = match self {
            Self::Spring(c) => (c.body_a, c.body_b, c.anchor_a, c.anchor_b),
            Self::Rod(c) => (c.body_a, c.body_b, c.anchor_a, c.anchor_b),
            Self::BallSocket(c) => (c.body_a, c.body_b, c.anchor_a, c.anchor_b),
            Self::Rope(c) => (c.body_a, c.body_b, c.anchor_a, c.anchor_b),
        };
        let body_a = bodies.get(body_a)?;
        let body_b = bodies.get(body_b)?;
        Some((
            body_a.position + body_a.orientation * anchor_a,
            body_b.position + body_b.orientation * anchor_b,
        ))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Spring(_) => "spring",
            Self::Rod(_) => "rod",
            Self::BallSocket(_) => "ball and socket",
            Self::Rope(_) => "rope",
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;

    #[test]
    fn endpoints_apply_body_rotation_and_local_anchors() {
        let bodies = vec![AvbdBody::cube(Vec3::ZERO, 0.5, 0.0), {
            let mut body = AvbdBody::cube(Vec3::new(2.0, 0.0, 0.0), 0.5, 1.0);
            body.orientation = Quat::from_rotation_z(90.0f32.to_radians());
            body
        }];
        let constraint = Constraint::BallSocket(BallSocket::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ));
        let (a, b) = constraint.endpoints(&bodies).expect("valid endpoints");
        assert_eq!(a, Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(b, Vec3::new(2.5, -0.0, 0.0));
    }

    #[test]
    fn constraint_names_cover_all_kinds() {
        let names = [
            Constraint::Spring(Spring::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0, 1.0)),
            Constraint::Rod(Rod::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0)),
            Constraint::BallSocket(BallSocket::new(0, 1, Vec3::ZERO, Vec3::ZERO)),
            Constraint::Rope(Rope::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0)),
        ]
        .map(|constraint| constraint.name());
        assert_eq!(names, ["spring", "rod", "ball and socket", "rope"]);
    }
}
