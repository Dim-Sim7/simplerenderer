#pragma once
#include "shader.h"

class TangentTextureShader : public IShader {
public:
    TangentTextureShader(const Model& model, const vec3& light);
    vec4 vertex(const int face, const int vert) override;
    Fragment fragment(const vec3 bar) const override;
    
private:
    const Model &model;
    vec3 l;              // light direction in view coordinates
    vec2  varying_uv[3]; // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    vec3 varying_norm[3]; // normal per vertex to be interpolated by the fragment shader
    vec3 varying_pos[3];
    vec4 tri[3];         // triangle in view coordinates
};