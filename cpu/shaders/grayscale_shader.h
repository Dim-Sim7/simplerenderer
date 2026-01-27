#pragma once
#include "shader.h"

class GrayscaleShader : public IShader {
public:
    GrayscaleShader(const Model& model, const vec3& light);
    vec4 vertex(const int face, const int vert) override;
    Fragment fragment(const vec3 bar) const override;

private:
    const Model &model;
    vec3 l;          // light direction in eye coordinates
    vec3 varying_nrm[3]; // normal per vertex to be interpolated by the fragment shader
};

