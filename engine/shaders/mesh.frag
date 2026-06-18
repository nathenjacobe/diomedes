#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_frag_pos;
layout(location = 2) in vec3 v_color;

layout(push_constant) uniform Push {
    vec4 camera_position; // xyz = camera position
    vec4 light_direction; // xyz = direction the light travels (points from source)
    vec4 light_color;     // xyz = light color
    vec4 light_params;    // (ambient, specular_power, specular_strength, _)
} push;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = -push.light_direction.xyz; // toward the light
    float diffuse = max(dot(n, l), 0.0);

    vec3 view_dir = normalize(push.camera_position.xyz - v_frag_pos);
    vec3 half_dir = normalize(l + view_dir);
    float specular = pow(max(dot(n, half_dir), 0.0), push.light_params.y);

    vec3 lit = v_color * (push.light_params.x + push.light_color.xyz * diffuse)
        + vec3(push.light_params.z * specular);
    out_color = vec4(lit, 1.0);
}
