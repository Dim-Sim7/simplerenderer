#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>

#include "./shaders/shader.h"
#include "graphics_library/graphics_library.h"
#include "graphics_library/model.h"
#include "graphics_library/gpu_material.h"

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>

//enum RenderMode : int {MODEL, POINTCLOUD};

struct RenderState {
    int width;
    int height;
    std::vector<uchar4> hostFramebuffer;
    TGAImage img;
    ShaderType current_shader;
    vec3 light_dir;
    std::vector<Triangle> cpu_tris;
    GPUTriangle* gpu_tris;
    size_t gpu_tris_size;
    std::vector<TileBin> bins;
    GPUTriangle* gpu_tris_buffer; // Pre-allocated pinned buffer

    TileRange* host_tileRanges;
    int* host_triangleIndexList;
    size_t maxTriangleIndices;
    float* host_zinit;
    uchar4* staging_buffer; // Pre-allocated staging buffer for textures
    size_t staging_capacity;
};

struct GPUScene {
    GPUTriangle* tris;
    TileRange* tileRanges;
    int* triangleIndexList;
    float* zbuffer;
    uchar4* framebuffer;
    GPUMaterial* material;
};

GPUScene uploadSceneToGPU(const RenderState& state, const Model& model, cudaStream_t stream) {
    GPUScene scene;

    // Upload triangles to GPU
    nvtxMarkA("Allocate and copy GPUTriangles to GPU");
    cudaMalloc(&scene.tris, state.gpu_tris_size * sizeof(GPUTriangle));
    cudaMemcpyAsync(scene.tris, state.gpu_tris, state.gpu_tris_size * sizeof(GPUTriangle),
                    cudaMemcpyHostToDevice, stream);

    // Prepare and upload tile ranges and triangle indices
    size_t numTiles = state.bins.size();
    size_t totalIndices = 0;
    for (const auto& bin : state.bins) {
        totalIndices += bin.triangleIndices.size();
    }

    if (totalIndices > state.maxTriangleIndices) {
        std::cerr << "Triangle index buffer overflow: " << totalIndices << " > " << state.maxTriangleIndices << std::endl;
        std::exit(1);
    }

    // Fill pre-allocated pinned memory
    int offset = 0;
    for (size_t i = 0; i < numTiles; ++i) {
        state.host_tileRanges[i] = {offset, (int)state.bins[i].triangleIndices.size()};
        memcpy(&state.host_triangleIndexList[offset], state.bins[i].triangleIndices.data(),
               state.bins[i].triangleIndices.size() * sizeof(int));
        offset += state.host_tileRanges[i].count;
    }

    // Upload tile ranges to GPU
    nvtxMarkA("Allocate and copy TileRanges to GPU");
    cudaMalloc(&scene.tileRanges, numTiles * sizeof(TileRange));
    cudaMemcpyAsync(scene.tileRanges, state.host_tileRanges, numTiles * sizeof(TileRange),
                    cudaMemcpyHostToDevice, stream);

    // Upload triangle index list to GPU
    nvtxMarkA("Allocate and copy TriangleIndexList to GPU");
    cudaMalloc(&scene.triangleIndexList, totalIndices * sizeof(int));
    cudaMemcpyAsync(scene.triangleIndexList, state.host_triangleIndexList, totalIndices * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Allocate Z-buffer (initialized later with kernel)
    nvtxMarkA("Allocate Z-buffer on GPU");
    cudaMalloc(&scene.zbuffer, state.width * state.height * sizeof(float));

    // Allocate and clear framebuffer
    nvtxMarkA("Allocate and clear framebuffer on GPU");
    cudaMalloc(&scene.framebuffer, state.width * state.height * sizeof(uchar4));
    cudaMemset(scene.framebuffer, 0, state.width * state.height * sizeof(uchar4));

    return scene;
}

__global__ void clear_z(float* z, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = y * width + x;
        z[i] = -INFINITY;
    }
}



void setup_scene(const int width, const int height) {
    constexpr vec3 eye    = {-0.5, 0, 2};
    constexpr vec3 center = {0, 0, 0};
    constexpr vec3 up     = {0, 1, 0};

    lookat(eye, center, up);
    init_perspective(length(eye - center));
    init_viewport(0, 0, width, height);
}

RenderState initialize_render_state(int width, int height) {
    const size_t MAX_TRIS = 100000; // Adjust based on expected model size
    RenderState state {
        .width = width,
        .height = height,
        .img = TGAImage(width, height, TGAImage::RGB),
        .current_shader = ShaderType::TEXTURE,
        .light_dir = normalize(vec3{-1, 1, 1})
    };

    // Pre-allocate pinned buffers for performance
    nvtxMarkA("Pre-allocate pinned buffer for triangles");
    cudaHostAlloc(&state.gpu_tris_buffer, MAX_TRIS * sizeof(GPUTriangle), cudaHostAllocDefault);

    int tilesX = (width + TILE_W - 1) / TILE_W;
    int tilesY = (height + TILE_H - 1) / TILE_H;
    size_t numTiles = tilesX * tilesY;
    size_t maxIndices = MAX_TRIS * 8;

    cudaHostAlloc(&state.host_tileRanges, numTiles * sizeof(TileRange), cudaHostAllocDefault);
    cudaHostAlloc(&state.host_triangleIndexList, maxIndices * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&state.host_zinit, width * height * sizeof(float), cudaHostAllocDefault);

    std::fill(state.host_zinit, state.host_zinit + width * height, -INFINITY);
    state.maxTriangleIndices = maxIndices;

    // Pre-allocate staging buffer for textures (32M texels should be enough for 4096x4096 textures)
    const size_t MAX_STAGING_TEXELS = 32 * 1024 * 1024;
    nvtxMarkA("Pre-allocate pinned staging buffer");
    cudaHostAlloc(&state.staging_buffer, MAX_STAGING_TEXELS * sizeof(uchar4), cudaHostAllocWriteCombined);
    state.staging_capacity = MAX_STAGING_TEXELS;

    return state;
}

void render_model(const Model& model, RenderState& state, int width, int height) {
    size_t nfaces = model.nfaces();
    const size_t MAX_TRIS = 100000;
    if (nfaces > MAX_TRIS) {
        std::cerr << "Model has too many faces: " << nfaces << " > " << MAX_TRIS << std::endl;
        return;
    }

    int tilesX = (width + TILE_W - 1) / TILE_W;
    int tilesY = (height + TILE_H - 1) / TILE_H;
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim(tilesX, tilesY);
    size_t numTiles = tilesX * tilesY;

    state.bins.clear();
    state.bins.resize(numTiles);

    // Prepare texture staging
    TextureUploadStaging staging;
    size_t totalTexels = 
        model.diffuse().width()  * model.diffuse().height() +
        model.normal().width()   * model.normal().height() +
        model.specular().width() * model.specular().height();
    if (totalTexels > state.staging_capacity) {
        std::cerr << "Model textures too large: " << totalTexels << " > " << state.staging_capacity << std::endl;
        return;
    }
    staging.buffer = state.staging_buffer;
    staging.capacity = state.staging_capacity;
    vec3 light_camera = (ModelView * vec4{state.light_dir.x, state.light_dir.y, state.light_dir.z, 0.0f}).xyz();
    // Process triangles
    init_triangles(model, state.cpu_tris);
    state.gpu_tris = state.gpu_tris_buffer;
    setupTrianglesCPU(model, state.cpu_tris, state.gpu_tris, state.gpu_tris_size);
    binTrianglesCPU(state.gpu_tris, state.gpu_tris_size, state.bins, width, height);

    // Render all shaders like CPU does
    std::cout << "ShaderType::COUNT = " << static_cast<int>(ShaderType::COUNT) << std::endl;

    for (int i = 0; i < static_cast<int>(ShaderType::COUNT); ++i) {
        ShaderType shaderType = static_cast<ShaderType>(i);
        state.current_shader = shaderType;

        // Clear framebuffer for this shader
        clear_framebuffer(state.img, TGAColor{255, 255, 22, 0});

        // Clear framebuffer for this shader
        // clear_framebuffer(state.img, TGAColor{255, 255, 22, 0});

        // Upload to GPU
        cudaStream_t uploadStream;
        cudaStreamCreate(&uploadStream);
        
        GPUMaterial h_mat = uploadMaterial(model, light_camera, staging, uploadStream);
        GPUMaterial* d_mat;
        cudaMalloc(&d_mat, sizeof(GPUMaterial));
        nvtxMarkA("Copy GPUMaterial to GPU");
        cudaMemcpyAsync(d_mat, &h_mat, sizeof(GPUMaterial), cudaMemcpyHostToDevice, uploadStream);
        GPUScene scene = uploadSceneToGPU(state, model, uploadStream);
        scene.material = d_mat;
        cudaStreamSynchronize(uploadStream);
        cudaStreamDestroy(uploadStream);

        // Render
        cudaStream_t rasterStream;
        cudaStreamCreate(&rasterStream);
        nvtxMarkA("Launch clear_z kernel");
        clear_z<<<gridDim, blockDim, 0, rasterStream>>>(scene.zbuffer, width, height);
        nvtxMarkA("Launch raster_GPU kernel");
        raster_GPU<<<gridDim, blockDim, 0, rasterStream>>>(
            scene.tris, scene.tileRanges, scene.triangleIndexList,
            tilesX, tilesY, width, height, scene.zbuffer, scene.framebuffer,
            state.current_shader, scene.material
        );

        write_to_framebuffer(state.hostFramebuffer, width, height, scene.framebuffer, state.img, rasterStream);
        cudaStreamDestroy(rasterStream);

        // Generate output filename and write
        std::string filename = std::string("framebuffer_") + shaderTypeToString(shaderType) + ".tga";
        std::cout << "WRITING TO FILE: " << filename << '\n';
        state.img.write_tga_file(filename.c_str());

        // Cleanup per-shader GPU resources
        cudaFree(scene.tris);
        cudaFree(scene.framebuffer);
        cudaFree(scene.material);
        cudaFree(scene.tileRanges);
        cudaFree(scene.triangleIndexList);
        cudaFree(scene.zbuffer);
    }
    // staging.buffer is pre-allocated, no need to free here
}

void cleanup_render_state(RenderState& state) {
    cudaFreeHost(state.gpu_tris_buffer);
    cudaFreeHost(state.host_tileRanges);
    cudaFreeHost(state.host_triangleIndexList);
    cudaFreeHost(state.host_zinit);
    cudaFreeHost(state.staging_buffer);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj [model2.obj ...]\n";
        return 1;
    }

    constexpr int width = 1024;
    constexpr int height = 1024;

    setup_scene(width, height);
    RenderState state = initialize_render_state(width, height);

    // Process each model
    for (int i = 1; i < argc; ++i) {
        Model model(argv[i]);
        auto start = std::chrono::high_resolution_clock::now();
        render_model(model, state, width, height);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "GPU rendering time for " << argv[i] << ": " << elapsed.count() << " seconds\n";
    }

    cleanup_render_state(state);
    return 0;
}