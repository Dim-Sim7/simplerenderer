#include "shader.h"
#include "../graphics_library/gpu_material.h"
#include <cuda_runtime.h>
#include "../math_lib/gpu_math.h"

__device__ uchar4 sample(const Texture2D& tex, const FragmentInput& fin) {
    float u = clampf(fin.uv.x, 0.0f, 1.0f);
    float v = clampf(fin.uv.y, 0.0f, 1.0f);

    //convert to pixel coords
    int x = int(u * (tex.width - 1));
    int y = int(v * (tex.height - 1));

    //row major indexing
    int idx = y * tex.width + x;

    return tex.data[idx];
}

__device__ FragmentOutput shade(ShaderType type,
                    const FragmentInput& in, const GPUMaterial& gpu_m) {
    
    FragmentOutput out;
    
    switch(type) {

        case GRAY: {
            float g = fabsf(in.normal.z);
            uint8_t c = (uint8_t)(g * 255.0f);
            out.color = make_uchar4(c, c, c, 255);
            break;
        }


        case TOON: {
            float l = fmaxf(0.0f, in.normal.z);
            float steps = 4.0f;
            float q = floorf(l * steps) / steps;
            uint8_t c = (uint8_t)(q * 255.0f);
            out.color = make_uchar4(c, c, c, 255);
            break;
        }

        case TEXTURE: {
            //sample tangent space normal
            uchar4 albedo = sample(gpu_m.diffuse, in);

            uchar4 nmap = sample(gpu_m.normal, in);

            float3 Nt = make_float3(
                nmap.x / 255.0f,
                nmap.y / 255.0f,
                nmap.z / 255.0f
            );

            Nt = Nt * 2.0f - 1.0f;
            Nt = normalize(Nt);
            
            
            float3 N = normalize(in.normal);
            float3 T = normalize(in.tangent - N * dot(N, in.tangent));

            //Transform normal from tangent space -> view / world space
            float3 Np = normalize(
                in.tangent * Nt.x +
                in.bitangent * Nt.y +
                in.normal * Nt.z
            );
                    
            float light = fmaxf(0.0f, dot(Np, normalize(gpu_m.lightDir)));

            out.color = make_uchar4(
                uint8_t(albedo.x * light),
                uint8_t(albedo.y * light),
                uint8_t(albedo.z * light),
                255
            );
            
            break;

        }
    }

    return out;
    
}

