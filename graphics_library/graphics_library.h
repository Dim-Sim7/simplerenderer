#pragma once

#include "tgaimage.h"
#include "model.h"
#include "./math_lib/matrix.h"

enum ShaderType : int;   // forward declaration

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int THREADS_PER_BLOCK = 256;

extern std::vector<float> h_zbuffer;
extern mat4 ModelView, Viewport, Perspective;

struct GPUMaterial;
struct Triangle {
    vec4 pos[3];    //clip or view space position
    vec2 uv[3];     //per-vertex uv
    vec3 normal[3]; //per-vertex normal
    vec4 clip[3];
};

// struct Point {
//     vec3 pos;
//     vec3 color;
//     float size;
// };

struct GPUPoint {
    float2 screen_pos;
    float3 color;
    float size;
    float depth;
};

struct GPUTriangle {
    //Edge functions: Ax + By + C >= 0
    float3 edges[3];
    //Depth plane: z = Ax + By + C
    float3 z;

    float3 invW;
    

    //bounding box (screen space)
    int bbminx, bbmaxx;
    int bbminy, bbmaxy;
    float area;

    float3 uv_u;
    float3 uv_v;

    float3 nx, ny, nz;
    float3 bx, by, bz;
    float3 tx, ty, tz;
};

struct PointBin {
    std::vector<int> pointIndices;
};

struct TileBin {
    std::vector<int> triangleIndices;
};

struct TileRange {
    int start;
    int count;
};

void lookat(const vec3 eye, const vec3 center, const vec3 up);
void init_perspective(const float f);
void init_viewport(const float x, const float y, const float w, const float h);
void init_zbuffer(const int width, const int height);


void init_triangles(const Model& model, std::vector<Triangle>& cpu_tris);

void setupTrianglesCPU(const Model& model, const std::vector<Triangle>& input, GPUTriangle* out, size_t& out_size);
void binTrianglesCPU(const GPUTriangle* tris, size_t& gpu_tris_size, std::vector<TileBin>& bins, int frame_width, int frame_height);


__global__ void raster_GPU(const GPUTriangle* tris, const TileRange* tileRanges, const int* triangleIndexList,
                            int tilesX, int tilesY, int screenW, int screenH, float* zbuffer, uchar4* framebuffer, ShaderType type, const GPUMaterial* material);

void write_to_framebuffer(std::vector<uchar4>& hostFramebuffer, int screenW, int screenH, const uchar4* d_framebuffer, TGAImage& img, cudaStream_t stream);

void clear_framebuffer(TGAImage& img, const TGAColor& color);




