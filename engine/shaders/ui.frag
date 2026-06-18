#version 450

// vertex colors are premultiplied srgb; the color attachment is srgb; the
// product stays in gamma space and blends with one / one_minus_src_alpha;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(set = 0, binding = 0) uniform sampler2D u_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(u_sampler, v_uv) * v_color;
}
