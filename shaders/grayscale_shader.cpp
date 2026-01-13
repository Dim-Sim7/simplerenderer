#include "grayscale_shader.h"

GrayscaleShader::GrayscaleShader(const Model& model, const vec3& light)
    : model(model) {
        l = normalize((ModelView * vec4{light.x, light.y, light.z, 0.0}).xyz());

}

vec4 GrayscaleShader::vertex(const int face, const int vert) {
    vec4 v = model.vert(face, vert);        //current vertex in object coords
    vec3 n = model.normal(face, vert).xyz();
    varying_nrm[vert] = (ModelView.invert_transpose() * vec4{n.x, n.y, n.z, 0.}).xyz();
    vec4 gl_Position = ModelView * v;
    return Perspective * gl_Position;       //clip coords

}

Fragment GrayscaleShader::fragment(const vec3 bar) const {
    TGAColor gl_FragColor = {255, 255, 255, 255};
    vec3 n = normalize(varying_nrm[0] * bar[0] +
                       varying_nrm[1] * bar[1] +
                       varying_nrm[2] * bar[2]); // per-vertex normal interpolation
    vec3 r = normalize(n * (2.0 * dot(n, l)) - l); // reflected light
    double ambient = 0.3;
    double diffuse = std::max(0.0, dot(n, l));
    double spec = std::pow(std::max(r.z, 0.0), 35);

    for (int channel: {0, 1, 2}) {
        gl_FragColor[channel] *= std::min(1.0, ambient + 0.4*diffuse + 0.9*spec);
    }
    return Fragment{false, gl_FragColor, n};

}