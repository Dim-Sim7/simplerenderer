#pragma once
#include <utility>
#include "../graphics_library/tgaimage.h"
#include "../graphics_library/model.h"
#include "../graphics_library/graphics_library.h"
#include "../graphics_library/gpu_material.h"
#include <vector_types.h>

enum ShaderType : int {TOON = 0, GRAY = 1, TEXTURE = 2, COUNT = 3};
enum RenderType : int {POINTCLOUD = 0, LINE = 1};

inline const char* shaderTypeToString(ShaderType type) {
    switch (type) {
        case ShaderType::TOON:        return "toon";
        case ShaderType::GRAY:        return "gray";
        case ShaderType::TEXTURE:     return "tangent";
        default:                      return "unknown";
    }
}

struct Fragment {
    bool discard;
    TGAColor color;
    vec3 normal;
};

struct FragmentInput {
    float2 uv;
    float3 normal;   //N
    float3 tangent;  //T
    float3 bitangent;//B
    float3 fragPos;
};

struct FragmentOutput {
    uchar4 color; // CUDA RGBA
};

__device__ uchar4 sample(const Texture2D& tex, const FragmentInput& fin);

__device__ FragmentOutput shade(ShaderType type,
                    const FragmentInput& in, const GPUMaterial& gpu_m);