use glam::Vec3;

/// a simple directional light with ambient and Blinn-Phong specular; set via
/// `crate::render::renderer::light_mut`; users can animate it as they so please
#[derive(Clone, Copy, Debug)]
pub struct Light {
    /// direction the light travels (points from the source), normalised
    pub direction: Vec3,
    /// light color as rgb intensity
    pub color: Vec3,
    /// ambient light level added to every surface
    pub ambient: f32,
    /// Blinn-Phong specular exponent
    pub specular_power: f32,
    /// strength of the specular highlight
    pub specular_strength: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.4, -0.7, 0.6).normalize(),
            color: Vec3::new(1.0, 0.97, 0.92),
            ambient: 0.22,
            specular_power: 32.0,
            specular_strength: 0.5,
        }
    }
}
