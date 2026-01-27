#include <algorithm>
#include "graphics_library.h"
#include "./shaders/shader.h"

#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>

#include "./math_lib/gpu_math.h"
#include "gpu_material.h"

#include <cuda_runtime.h>

mat4 ModelView, Viewport, Perspective;
std::vector<float> h_zbuffer;
//rotate and translate the world so the camera is at the origin and looks down z"
void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 z = normalize(eye - center);      // camera forward
    vec3 x = normalize(cross(up, z));      // camera right
    vec3 y = cross(z, x);                  // camera up

    //express world coords in camera axes
    mat4 R = {{
        { x.x, x.y, x.z, 0 },
        { y.x, y.y, y.z, 0 },
        { z.x, z.y, z.z, 0 },
        { 0,   0,   0,   1 }
    }};

    //translate the world so that center move to origin
    mat4 T = {{
        { 1, 0, 0, -center.x },
        { 0, 1, 0, -center.y },
        { 0, 0, 1, -center.z },
        { 0, 0, 0, 1 }
    }};

    //translate then rotate world
    ModelView = R * T;
}


void init_perspective(const float f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1.0f/f,1}}};
}

void init_viewport(const float x, const float y, const float w, const float h) {
    Viewport = {{{w/2.0f, 0.0f, 0.0f, x+w/2.0f}, {0.0f, h/2.0f, 0.0f, y+h/2.0f}, {0.0f,0.0f,1.0f,0.0f}, {0.0f,0.0f,0.0f,1.0f}}};
}

void init_zbuffer(const int width, const int height) {
    h_zbuffer = std::vector(width * height, -std::numeric_limits<float>::infinity());
}


inline float signed_area(vec2 screen[3]) {
    return (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) - (screen[1].y - screen[0].y) * (screen[2].x - screen[0].x);
}

void init_triangles(const Model& model, std::vector<Triangle>& cpu_tris) {

    cpu_tris.clear();
    cpu_tris.reserve(model.nfaces());
    for (int f = 0; f < model.nfaces(); ++f) {
        Triangle tri;
        for (int v = 0; v < 3; ++v) {
            //object space pos
            tri.pos[v] = model.vert(f, v);
            tri.clip[v] = Perspective * ModelView * tri.pos[v];
            
            //attributes
            tri.uv[v] = model.uv(f, v);
            vec3 n_obj = model.normal(f, v).xyz();
            vec3 n_view = (ModelView * vec4{n_obj.x, n_obj.y, n_obj.z, 0.0f}).xyz();
            tri.normal[v] = normalize(n_view);
        }

        cpu_tris.push_back(tri);
    }
}

inline float3 make_plane(const float a[3], const float3 edges[3], float area) {
    float3 p{0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i) {
        p.x += a[i] * edges[i].x;
        p.y += a[i] * edges[i].y;
        p.z += a[i] * edges[i].z;
    }
    return {p.x / area, p.y / area, p.z / area};
}

void setupPointsCPU(const Model& model, const std::vector<Triangle>& input, GPUPoint* out, size_t& out_size) {
    size_t index = 0;
    for (const Triangle& tri : input) {
        vec4 ndc[3] = { tri.clip[0]/tri.clip[0].w, tri.clip[1]/tri.clip[1].w, tri.clip[2]/tri.clip[2].w };
        vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates
        
        float area = signed_area(screen);
        if (area < 1e-4) continue;
        auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
        auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners

        GPUPoint point;
        float cx = (bbminx + bbmaxx) * 0.5f;
        float cy = (bbminy + bbmaxy) * 0.5f;

        point.screen_pos.x = cx;
        point.screen_pos.y = cy;

        point.size = 8.0f; //8 pixels width

        float zvals[3] = {ndc[0].z, ndc[1].z, ndc[2].z};

        //interpolate depth at centre using barycentric coords
        point.depth = (zvals[0] + zvals[1] + zvals[2]) / 3.0f;

        point.color = {1.0f, 1.0f, 1.0f};

        out[index++] = point;

    }
    out_size = index;
}

void binPointsCPU(const GPUPoint* points, size_t& points_size, std::vector<PointBin>& bins, int frame_width, int frame_height) {
    int tilesX = (frame_width + TILE_W - 1) / TILE_W;
    int tilesY = (frame_height + TILE_H - 1) / TILE_H;
    int num_tiles = tilesX * tilesY;
    bins.clear();
    bins.resize(num_tiles);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<int>>> local_bins(num_threads, std::vector<std::vector<int>>(num_tiles));
    //add point to correct tilebin
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
#pragma omp for
        for (size_t i = 0; i < points_size; ++i) {

            //Compute tile coords from screen position
            int tileX = (int)(points[i].screen_pos.x / TILE_W);
            int tileY = (int)(points[i].screen_pos.y / TILE_H);

            //Clamp to valid range
            tileX = std::clamp(tileX, 0, tilesX - 1);
            tileY = std::clamp(tileY, 0, tilesY - 1);

            //Compute tileID
            int tileID = tileY * tilesX + tileX;

            //Add points index to the local bin for this tile
            local_bins[thread_id][tileID].push_back(i);
        }
    }

    // Merge local bins into global bins
    // for every thread, for every tile, merge that threads data into the global tile bin
    for (int t = 0; t < num_threads; ++t) {
        for (int tileID = 0; tileID < num_tiles; ++tileID) {
            bins[tileID].pointIndices.insert(
                bins[tileID].pointIndices.end(),
                local_bins[t][tileID].begin(),
                local_bins[t][tileID].end()
            );
        }
    }

}

// __global__ void raster_points_GPU(const GPUPoint* points, const TileRange* tileRanges, const int* pointIndexList,
//                                     int tilesX, int tilesY, int screenW, int screenH, float* d_zbuffer, uchar4* framebuffer)

void setupTrianglesCPU(const Model& model, const std::vector<Triangle>& input, GPUTriangle* out, size_t& out_size) {
    size_t index = 0;
    for (const Triangle& tri : input) {
        vec4 ndc[3] = { tri.clip[0]/tri.clip[0].w, tri.clip[1]/tri.clip[1].w, tri.clip[2]/tri.clip[2].w };
        vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates


        float area = signed_area(screen);
        if (area < 1e-4) continue;
        auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
        auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners

 
        GPUTriangle gpu;
        //Compute gpu triangle points e0.x = e1.y - e2.y ...
        gpu.area = area;
        gpu.edges[0].x = screen[1].y - screen[2].y;
        gpu.edges[0].y = screen[2].x - screen[1].x;
        gpu.edges[0].z = (screen[1].x*screen[2].y) - (screen[2].x*screen[1].y);

        gpu.edges[1].x = screen[2].y - screen[0].y;
        gpu.edges[1].y = screen[0].x - screen[2].x;
        gpu.edges[1].z = (screen[2].x*screen[0].y) - (screen[0].x*screen[2].y);

        gpu.edges[2].x = screen[0].y - screen[1].y;
        gpu.edges[2].y = screen[1].x - screen[0].x;
        gpu.edges[2].z = (screen[0].x*screen[1].y) - (screen[1].x*screen[0].y);

        for (int i = 0; i < 3; ++i) {
            float test =       
                gpu.edges[i].x * screen[i].x +
                gpu.edges[i].y * screen[i].y +
                gpu.edges[i].z;

            assert(test >= 0);
        }
        float zvals[3] = {ndc[0].z, ndc[1].z, ndc[2].z};
        gpu.z = make_plane(zvals, gpu.edges, area);

        for (int i = 0; i < 3; ++i) {
            float z =
            gpu.z.x * screen[i].x +
            gpu.z.y * screen[i].y +
            gpu.z.z;

            assert(abs(z - ndc[i].z) < 1e-4f);
        }

        gpu.bbmaxx = bbmaxx;
        gpu.bbmaxy = bbmaxy;
        gpu.bbminx = bbminx;
        gpu.bbminy = bbminy;

        float invW[3] = {
            1.0f / tri.clip[0].w,
            1.0f / tri.clip[1].w,
            1.0f / tri.clip[2].w
        };

        gpu.invW = make_plane(invW, gpu.edges, area);

        float u[3] = {tri.uv[0].x * invW[0], tri.uv[1].x * invW[1], tri.uv[2].x * invW[2]};
        float v[3] = {tri.uv[0].y * invW[0], tri.uv[1].y * invW[1], tri.uv[2].y * invW[2]};


        gpu.uv_u = make_plane(u, gpu.edges, area);
        gpu.uv_v = make_plane(v, gpu.edges, area);

        float nx[3] = {tri.normal[0].x * invW[0], tri.normal[1].x * invW[1], tri.normal[2].x * invW[2]};
        float ny[3] = {tri.normal[0].y * invW[0], tri.normal[1].y * invW[1], tri.normal[2].y * invW[2]};
        float nz[3] = {tri.normal[0].z * invW[0], tri.normal[1].z * invW[1], tri.normal[2].z * invW[2]};

        gpu.nx = make_plane(nx, gpu.edges, area);
        gpu.ny = make_plane(ny, gpu.edges, area);
        gpu.nz = make_plane(nz, gpu.edges, area);
        
        //Triangle edges in view-space space
        vec3 e1 = tri.pos[1].xyz() - tri.pos[0].xyz();
        vec3 e2 = tri.pos[2].xyz() - tri.pos[0].xyz();

        //Triangle edges UV space
        vec2 duv1 = tri.uv[1] - tri.uv[0];
        vec2 duv2 = tri.uv[2] - tri.uv[0];

        
        float det = duv1.x * duv2.y - duv1.y * duv2.x;
        if (fabs(det) < 1e-8f) continue;
        float invDet = 1.0f / det;

        vec3 T = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
        //vec3 B = normalize((-1.0*e1 * duv2.x + e2 * duv1.x) * invDet);

        vec3 Tvert[3];
        vec3 Bvert[3];

        for (int i = 0; i < 3; ++i) {
            vec3 N = normalize(tri.normal[i]);

            // Remove normal component from tangent
            vec3 Ti = T - N * dot(N, T);
            Ti = normalize(Ti);

            vec3 Bi = cross(N, Ti);

            Tvert[i] = Ti;
            Bvert[i] = Bi;
        }
        float tx[3], ty[3], tz[3];
        float bx[3], by[3], bz[3];

        for (int i = 0; i < 3; ++i) {
            tx[i] = Tvert[i].x * invW[i];
            ty[i] = Tvert[i].y * invW[i];
            tz[i] = Tvert[i].z * invW[i];

            bx[i] = Bvert[i].x * invW[i];
            by[i] = Bvert[i].y * invW[i];
            bz[i] = Bvert[i].z * invW[i];
        }
        gpu.tx = make_plane(tx, gpu.edges, area);
        gpu.ty = make_plane(ty, gpu.edges, area);
        gpu.tz = make_plane(tz, gpu.edges, area);

        gpu.bx = make_plane(bx, gpu.edges, area);
        gpu.by = make_plane(by, gpu.edges, area);
        gpu.bz = make_plane(bz, gpu.edges, area);



        out[index++] = gpu;

    }
    out_size = index;
}



void binTrianglesCPU(const GPUTriangle* tris, size_t& tris_size, std::vector<TileBin>& bins, int frame_width, int frame_height) {

    //define grid dimensions (how many cuda blocks to use) ---- gridDim = (tilesX, tilesY)
    int tilesX = (frame_width + TILE_W - 1) / TILE_W;
    int tilesY = (frame_height + TILE_H - 1) / TILE_H;
    int num_tiles = tilesX * tilesY;
    bins.clear();
    bins.resize(num_tiles);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<int>>> local_bins(num_threads, std::vector<std::vector<int>>(num_tiles));

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
#pragma omp for
        for (size_t i = 0; i < tris_size; ++i) {
            //convert bounding box to tile space
            int tileMinX = tris[i].bbminx / TILE_W;
            int tileMinY = tris[i].bbminy / TILE_H;
            int tileMaxX = tris[i].bbmaxx / TILE_W;
            int tileMaxY = tris[i].bbmaxy / TILE_H;

            tileMinX = std::clamp(tileMinX, 0, tilesX - 1);
            tileMinY = std::clamp(tileMinY, 0, tilesY - 1);
            tileMaxX = std::clamp(tileMaxX, 0, tilesX - 1);
            tileMaxY = std::clamp(tileMaxY, 0, tilesY - 1);

            //loop over the tile-space bounding box of the triangle and add it to their respective bins
            for (int x = tileMinX; x <= tileMaxX; ++x) {
                for (int y = tileMinY; y <= tileMaxY; ++y) {
                    //down y rows * tilesX bins per row + x (total offset)
                    //matches GPU Cuda kernel (blockIdx...)
                    int tileID = y * tilesX + x;
                    //each bins[tileID] contains a list of triangle indices whose bb overlap with that tile
                    local_bins[thread_id][tileID].push_back(i);
                }
            }
        }
    }

    // Merge local bins into global bins
    for (int t = 0; t < num_threads; ++t) {
        for (int tileID = 0; tileID < num_tiles; ++tileID) {
            bins[tileID].triangleIndices.insert(
                bins[tileID].triangleIndices.end(),
                local_bins[t][tileID].begin(),
                local_bins[t][tileID].end()
            );
        }
    }



}

__global__ void raster_GPU(const GPUTriangle* tris, const TileRange* tileRanges, const int* triangleIndexList,
                            int tilesX, int tilesY, int screenW, int screenH, float* d_zbuffer, uchar4* framebuffer, ShaderType type, const GPUMaterial* material) {
    __shared__ float tileZbuffer[TILE_W * TILE_H];
    
    //identify which tile this block is responsible for
    int tileID = blockIdx.y * tilesX + blockIdx.x;   
    //calculate pixel coords of this thread
    float pixelX = blockIdx.x * TILE_W + threadIdx.x;
    float pixelY = blockIdx.y * TILE_H + threadIdx.y;
    int localIdx = threadIdx.y * TILE_W + threadIdx.x;
    bool inBounds = (pixelX < screenW && pixelY < screenH);
    int idx = inBounds ? (pixelX + pixelY * screenW) : 0;  // dummy if not in bounds
    
    // Load global zbuffer into shared memory
    tileZbuffer[localIdx] = inBounds ? d_zbuffer[idx] : -1e20f;
    //triangle list for this tile
    TileRange range = tileRanges[tileID];
    float bestZ = tileZbuffer[localIdx];
    int bestTri = -1;

    //Iterate through all triangles at this pixel
    //Visibility test
    for (int i = 0; i < range.count; ++i) {
        int triIndex = triangleIndexList[range.start + i];
        const GPUTriangle& tri = tris[triIndex];
        //Bounding box test
        if ((pixelX < tri.bbminx || pixelX > tri.bbmaxx) || (pixelY < tri.bbminy || pixelY > tri.bbmaxy)) continue;

        //half-space edge function test
        float e0 = tri.edges[0].x * pixelX + tri.edges[0].y * pixelY + tri.edges[0].z;
        float e1 = tri.edges[1].x * pixelX + tri.edges[1].y * pixelY + tri.edges[1].z;
        float e2 = tri.edges[2].x * pixelX + tri.edges[2].y * pixelY + tri.edges[2].z;

        if (e0 < 0 || e1 < 0 || e2 < 0) continue;

        //Depth test
        float z = tri.z.x * pixelX + tri.z.y * pixelY + tri.z.z;
        
        if (z > bestZ) {
            bestZ = z;
            bestTri = triIndex;
        }

    }

    if (bestTri >= 0) {
        //Depth write
        tileZbuffer[localIdx] = bestZ;

        const GPUTriangle& tri = tris[bestTri];
        FragmentInput fin;
        float invW = tri.invW.x * pixelX +
                     tri.invW.y * pixelY +
                     tri.invW.z;
        //Interpolate UVs
        {
        float u =  (tri.uv_u.x * pixelX + tri.uv_u.y * pixelY + tri.uv_u.z) / invW;
        float v =  (tri.uv_v.x * pixelX + tri.uv_v.y * pixelY + tri.uv_v.z) / invW;
        fin.uv = make_float2(u, v);
        }

        //Interpolate Normals
        {
        float nx = (tri.nx.x * pixelX + tri.nx.y * pixelY + tri.nx.z) / invW;
        float ny = (tri.ny.x * pixelX + tri.ny.y * pixelY + tri.ny.z) / invW;
        float nz = (tri.nz.x * pixelX + tri.nz.y * pixelY + tri.nz.z) / invW;

        float3 N = normalize(float3{nx, ny, nz});
        fin.normal = N;
        }
        {
        float3 T = normalize({
        (tri.tx.x * pixelX + tri.tx.y * pixelY + tri.tx.z) / invW,
        (tri.ty.x * pixelX + tri.ty.y * pixelY + tri.ty.z) / invW,
        (tri.tz.x * pixelX + tri.tz.y * pixelY + tri.tz.z) / invW
        });
        T = normalize(T - fin.normal * dot(fin.normal, T));
        fin.tangent = T;
        }
        
        //Fragment shading
        {
        float3 B = normalize(cross(fin.normal, fin.tangent));
        fin.bitangent = B;
        }

        //Final pixel colour for this thread
        FragmentOutput fout = shade(type, fin, *material);
        //Store colour in framebuffer
        framebuffer[idx] = fout.color;
    }

    __syncthreads();
    if (inBounds) {
        d_zbuffer[idx] = tileZbuffer[localIdx];
    }

}

void write_to_framebuffer(std::vector<uchar4>& hostFramebuffer, int screenW, int screenH, const uchar4* d_framebuffer, TGAImage& img, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    size_t pixelCount = screenW * screenH;
    hostFramebuffer.resize(pixelCount);
    nvtxMarkA("Copy framebuffer to host");
    cudaMemcpy(
        hostFramebuffer.data(),
        d_framebuffer,
        screenW * screenH * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    );
    cudaDeviceSynchronize();
    for (int y = 0; y < screenH; ++y) {
        for (int x = 0; x < screenW; ++x) {
            uchar4 c = hostFramebuffer[x + y * screenW];
            img.set(x, y, TGAColor{c.z, c.y, c.x, c.w});
        }
    }

    // Don't write file here - let caller do it
}


void clear_framebuffer(TGAImage& img, const TGAColor& color) {
    for (int x = 0; x < img.width(); ++x) {
        for (int y = 0; y < img.height(); ++y) {
            img.set(x, y, color);
        }
    }
}