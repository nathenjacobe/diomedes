//! text rendering is really, really hard and complicated so im just going to use egui

mod pipeline;
mod renderer;
mod texture;

pub use renderer::UiRenderer;

use egui::epaint::textures::TexturesDelta;

/// tessellated ui frame: what to paint, plus pending texture changes
pub struct UiFrame {
    pub primitives: Vec<egui::epaint::ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub pixels_per_point: f32,
}

impl UiFrame {
    /// tessellate the egui output into renderable primitives
    pub fn new(
        ctx: &egui::Context,
        shapes: Vec<egui::epaint::ClippedShape>,
        textures_delta: TexturesDelta,
        pixels_per_point: f32,
    ) -> Self {
        let primitives = ctx.tessellate(shapes, pixels_per_point);
        Self {
            primitives,
            textures_delta,
            pixels_per_point,
        }
    }
}
