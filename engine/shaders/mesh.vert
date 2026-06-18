#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

// one (mvp, model) pair per scene instance; interleaved exactly like the
// cpu-side uniform buffer; indexed by gl_instanceindex, which includes the
// draw's firstinstance; each shape's batched draw addresses its own block;
// the array size must match max_instances in render/context;rs;
struct InstanceTransform {
    mat4 mvp;
    mat4 model;
};
layout(set = 0, binding = 0) uniform UniformBufferObject {
    InstanceTransform transforms[256];
} ubo;

layout(push_constant) uniform Push {
    vec4 camera_position; // xyz = camera position;
    vec4 light_direction; // xyz = direction the light travels; points from source;
    vec4 light_color;     // xyz = light color;
    vec4 light_params;    // ambient; specular_power; specular_strength; _;
} push;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_frag_pos;
layout(location = 2) out vec3 v_color;

void main() {
    InstanceTransform transform = ubo.transforms[gl_InstanceIndex];
    vec4 world = transform.model * vec4(position, 1.0);
    v_frag_pos = world.xyz;
    v_normal = mat3(transform.model) * normal;
    v_color = color;
    gl_Position = transform.mvp * vec4(position, 1.0);
}
