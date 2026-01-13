#pragma once
#include "tgaimage.h"
#include "geometry.h"
#include "shaders/shader.h"



void lookat(const vec3 eye, const vec3 center, const vec3 up);
void init_perspective(const double f);
void init_viewport(const int x, const int y, const int w, const int h);
void init_zbuffer(const int width, const int height);
void init_normalBuffer(const int width, const int height);
typedef vec4 Triangle[3]; // a triangle primitive is made of three ordered points
void rasterize(const Triangle &clip, const IShader &shader, TGAImage &framebuffer);

void clear_framebuffer(TGAImage& img, const TGAColor& color);