#include "gpu_material.h"
#include <nvtx3/nvToolsExt.h>

GPUMaterial uploadMaterial(const Model& model, const vec3& lightDir, TextureUploadStaging& staging, cudaStream_t stream) {
    GPUMaterial mat{};
    
    // Calculate offsets in the staging buffer for each texture
    size_t diffuseOffset = 0;
    size_t normalOffset = model.diffuse().width() * model.diffuse().height();
    size_t specularOffset = normalOffset + model.normal().width() * model.normal().height();
    
    // Diffuse
    const TGAImage& diff = model.diffuse();
    size_t diffSize = diff.width() * diff.height();
    uchar4* diffuseBuffer = staging.buffer + diffuseOffset;

    for (int y = 0; y < diff.height(); ++y) {
        for (int x = 0; x < diff.width(); ++x) {
            TGAColor c = diff.get(x, y);
            diffuseBuffer[y * diff.width() + x] = make_uchar4(
                c[2], // B
                c[1], // G
                c[0], // R
                c[3]  // A
            );
        }
    }

    cudaMalloc(&mat.diffuse.data, diffSize * sizeof(uchar4));
    nvtxMarkA("Copy diffuse texture to GPU");
    cudaMemcpyAsync(mat.diffuse.data, diffuseBuffer,
            diffSize * sizeof(uchar4),
            cudaMemcpyHostToDevice, stream);

    mat.diffuse.width = diff.width();
    mat.diffuse.height = diff.height();
  

    // Normal
    const TGAImage& norm = model.normal();
    size_t normSize = norm.width() * norm.height();
    uchar4* normalBuffer = staging.buffer + normalOffset;

    for (int y = 0; y < norm.height(); ++y) {
        for (int x = 0; x < norm.width(); ++x) {
            TGAColor c = norm.get(x, y);
            normalBuffer[y * norm.width() + x] = make_uchar4(
                c[2],
                c[1],
                c[0],
                c[3]
            );
        }
    }

    cudaMalloc(&mat.normal.data, normSize * sizeof(uchar4));
    nvtxMarkA("Copy normal texture to GPU");
    cudaMemcpyAsync(mat.normal.data, normalBuffer, normSize * sizeof(uchar4), cudaMemcpyHostToDevice, stream);

    mat.normal.width = norm.width();
    mat.normal.height = norm.height();

    // Specular
    const TGAImage& spec = model.specular();
    size_t specSize = spec.width() * spec.height();
    uchar4* specularBuffer = staging.buffer + specularOffset;

    for (int y = 0; y < spec.height(); ++y) {
        for (int x = 0; x < spec.width(); ++x) {
            TGAColor c = spec.get(x, y);
            specularBuffer[y * spec.width() + x] = make_uchar4(
                c[2],
                c[1],
                c[0],
                c[3]
            );
        }
    }

    cudaMalloc(&mat.specular.data, specSize * sizeof(uchar4));
    nvtxMarkA("Copy specular texture to GPU");
    cudaMemcpyAsync(mat.specular.data, specularBuffer, specSize * sizeof(uchar4), cudaMemcpyHostToDevice, stream);

    mat.specular.width = spec.width();
    mat.specular.height = spec.height();

    // Light
    mat.lightDir = make_float3(lightDir.x, lightDir.y, lightDir.z);

    return mat;
}



