#pragma once
#include "shader.h"


class ToonShader : public IShader {
public:
    ToonShader(const Model& model, const vec3& light, const TGAColor& c);
    
    vec4 vertex(const int face, const int vert) override;
    Fragment fragment(const vec3 bar) const override;

private:
    const Model& model;
    TGAColor color;
    vec3 l;                 //light direction in eye coords
    vec4 varying_nrm[3];   // normal per vertex
};