#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target[2048];
layout(set = 0, binding = 1) uniform texture2D t_Color;
layout(set = 0, binding = 2) uniform sampler s_Color;
layout(push_constant) uniform PushConstant{
    int index;
} pc;
void main() {
    vec4 tex = texture(
        sampler2D(t_Color, s_Color),
        vec2(v_TexCoord.x, v_TexCoord.y)
    );
    float mag = length(v_TexCoord-vec2(0.5));
    o_Target[pc.index] = vec4(mix(tex.xyz, vec3(0.0), mag*mag), 1.0);
}
