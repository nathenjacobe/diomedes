#version 450

// egui screen-space rendering; positions arrive in physical pixels with
// (0,0) at the top-left; colors are premultiplied srgbA normalized from an
// r8g8b8a8_unorm vertex attribute;

// egui's coordinate system is already y-down, matching vulkan ndc; no
// vertical flip is applied here;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(push_constant) uniform Push {
    vec2 screen_size; // physical pixels;
} push;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    v_uv = a_uv;
    v_color = a_color;
    vec2 clip = vec2(
        2.0 * a_pos.x / push.screen_size.x - 1.0,
        2.0 * a_pos.y / push.screen_size.y - 1.0
    );
    gl_Position = vec4(clip, 0.0, 1.0);
}
