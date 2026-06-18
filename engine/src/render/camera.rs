use glam::camera::rh::proj::directx;
use glam::camera::rh::view::look_at_mat4;
use glam::{Mat4, Vec3};

/// perspective camera positioned with an eye/target/up trio
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y: f32,
    z_near: f32,
    z_far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self::look_at(Vec3::new(2.5, 2.5, 2.5), Vec3::ZERO, Vec3::Y)
    }
}

impl Camera {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self {
            eye,
            target,
            up,
            fov_y: 45.0f32.to_radians(),
            z_near: 0.1,
            z_far: 100.0,
        }
    }

    /// view matrix: world space to camera space
    pub fn view(&self) -> Mat4 {
        look_at_mat4(self.eye, self.target, self.up)
    }

    /// unit vector from the eye toward the target (the look direction)
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye).normalize()
    }

    /// camera-local right vector, perpendicular to the look direction and up
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// move the camera along its local axes: positive `right` strafes right,
    /// `forward` moves toward the target, `up` moves along the camera's up
    /// vector; the look direction is preserved
    pub fn move_local(&mut self, right: f32, forward: f32, up: f32) {
        let delta = self.right() * right + self.forward() * forward + self.up * up;
        self.eye += delta;
        self.target += delta;
    }

    /// projection matrix in vulkan clip conventions: y flipped down (ndc y
    /// grows downward) and depth in [0, 1]
    pub fn projection(&self, aspect_ratio: f32) -> Mat4 {
        let projection = directx::perspective(self.fov_y, aspect_ratio, self.z_near, self.z_far);
        Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0)) * projection
    }

    pub fn position(&self) -> Vec3 {
        self.eye
    }

    pub fn set_position(&mut self, eye: Vec3) {
        self.eye = eye;
    }

    pub fn target(&self) -> Vec3 {
        self.target
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn set_up(&mut self, up: Vec3) {
        self.up = up;
    }

    pub fn set_vertical_fov(&mut self, fov_y: f32) {
        self.fov_y = fov_y;
    }

    /// the vertical field of view in radians
    pub fn vertical_fov(&self) -> f32 {
        self.fov_y
    }

    pub fn set_clip_planes(&mut self, z_near: f32, z_far: f32) {
        self.z_near = z_near;
        self.z_far = z_far;
    }
}
