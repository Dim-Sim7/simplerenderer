#pragma once
#include "model.h"
#include <cuda_runtime.h>

struct Texture2D {
    int width;
    int height;
    uchar4* data; //device pointer
};

struct GPUMaterial {
    Texture2D diffuse;
    Texture2D normal;
    Texture2D specular;

    float3 lightDir;
};

struct TextureUploadStaging {
    uchar4* buffer = nullptr;
    size_t capacity = 0;
};
GPUMaterial uploadMaterial(const Model& model, const vec3& lightDir, TextureUploadStaging& staging, cudaStream_t stream);