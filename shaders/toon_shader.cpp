#include "toon_shader.h"

ToonShader::ToonShader(const Model& model, const vec3& light, const TGAColor& c)
    : model(model), color(c) {
        l = normalize(ModelView * vec4{light.x, light.y, light.z, 0.0}).xyz();
    }

vec4 ToonShader::vertex(const int face, const int vert) {
        varying_nrm[vert] = ModelView.invert_transpose() * model.normal(face, vert);
        vec4 gl_Position = ModelView * model.vert(face, vert);
        return Perspective * gl_Position;
    }

Fragment ToonShader::fragment(const vec3 bar) const {
    vec3 n = normalize((varying_nrm[0] * bar[0] + varying_nrm[1] * bar[1] + varying_nrm[2] * bar[2]).xyz()); // per-vertex normal interpolation
    double diffuse = std::max(0.0, dot(n, l)); //diffuse light intensity
    double intensity = 0.15 + diffuse; //ambient + diffuse
    intensity = std::clamp(intensity, 0.0, 1.0);
    intensity = std::ceil(intensity * 3.0) / 3.0; //0.33, 0.66 or 1.0

    TGAColor gl_FragColor;
    for (int channel : {0, 1, 2})
        gl_FragColor[channel] = std::min<int>(255, color[channel] * intensity);
    return Fragment{false, gl_FragColor, n};
}
