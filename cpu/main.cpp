#include <cstdlib>
#include <iostream>
#include "graphics_library/graphics_library.h"
#include "graphics_library/model.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <omp.h>
#include <memory>
#include <chrono>
//Shaders
#include "./shaders/shader.h"
#include "./shaders/toon_shader.h"
#include "./shaders/grayscale_shader.h"
#include "./shaders/tangent_texture_shader.h"

extern mat4 ModelView, Viewport, Perspective;     // "OpenGL" state matrices and
extern std::vector<vec3> normalBuffer;
extern std::vector<double> zbuffer;

struct RenderState {
    int width;
    int height;
    TGAImage framebuffer;
    ShaderType current_shader;
    vec3 light_dir;
};


constexpr double golden_angle = 2.399963229728653; // radians

const char* shaderTypeToString(ShaderType type) {
    switch (type) {
        case ShaderType::TOON:        return "toon";
        case ShaderType::GRAY:        return "gray";
        case ShaderType::TEXTURE:     return "tangent";
        default:                      return "unknown";
    }
}

vec2 sample_disk(int i, int N) {
    double r = std::sqrt((i + 0.5) / N);
    double theta = i * golden_angle;
    return vec2{
        r * std::cos(theta),
        r * std::sin(theta)
    };

}

void ambient_occlusion_pass(RenderState& state) {
    constexpr double AO_RADIUS = 0.6;
    constexpr int AO_SAMPLES = 128;
    constexpr double AO_BIAS = 0.02;

    mat4 inv = (Viewport * Perspective * ModelView).invert();

#pragma omp parallel for
    for (int x = 0; x < state.width; ++x) {
        for (int y = 0; y < state.height; ++y) {
            double depth = zbuffer[x + y * state.width];
            if (depth < -100) continue;

            vec4 viewPixel = inv * vec4{
                static_cast<double>(x),
                static_cast<double>(y),
                depth,
                1.0
            };
            vec3 P = (viewPixel / viewPixel.w).xyz();

            vec3 N = normalize(normalBuffer[x + y * state.width]);

            vec3 up = std::abs(N.z) < 0.999 ? vec3{0,0,1} : vec3{0,1,0};
            vec3 T = normalize(cross(up, N));
            vec3 B = cross(N, T);

            double occlusion = 0.0;

            for (int i = 0; i < AO_SAMPLES; ++i) {
                vec2 d = sample_disk(i, AO_SAMPLES);
                double h = sqrt(std::max(0.0, 1.0 - d.x*d.x - d.y*d.y));

                vec3 dir_tangent = normalize(vec3{ d.x, d.y, h });
                vec3 dir = normalize(
                    T * dir_tangent.x +
                    B * dir_tangent.y +
                    N * dir_tangent.z
                );

                double radius = AO_RADIUS * std::abs(P.z);
                vec3 samplePos = P + dir * radius;

                vec4 clip = (Perspective * ModelView) * vec4{
                    samplePos.x, samplePos.y, samplePos.z, 1.0
                };
                if (clip.w <= 0.0) continue;

                vec3 ndc = clip.xyz() / clip.w;
                int sx = int((ndc.x * 0.5 + 0.5) * state.width);
                int sy = int((ndc.y * 0.5 + 0.5) * state.height);
                if (sx < 0 || sx >= state.width || sy < 0 || sy >= state.height)
                    continue;

                double sampleDepth = zbuffer[sx + sy * state.width];
                if (sampleDepth < -100) continue;

                vec4 sampleView = inv * vec4{
                    static_cast<double>(sx),
                    static_cast<double>(sy),
                    sampleDepth,
                    1.0
                };
                double sceneZ = (sampleView / sampleView.w).z;

                if (sceneZ < samplePos.z - AO_BIAS){
                    double range = radius;
                    double dist  = std::abs(sceneZ - samplePos.z);
                    double weight = 1.0 - std::clamp(dist / range, 0.0, 1.0);
                    occlusion += weight;
                }

            }
            double ao = 1.0 - occlusion / AO_SAMPLES;
            //ao = std::pow(std::clamp(ao, 0.0, 1.0), 1.5); // contrast
            ao = std::clamp(ao, 0.0, 1.0);
            ao = smoothstep(0.5, 0.95, ao);
            ao = std::pow(ao, 1.5);

            //std::cout << ao << '\n';
            TGAColor c = state.framebuffer.get(x, y);
            auto mod = [&](int v) {
                return uint8_t(std::clamp(v * ao, 0.0, 255.0));
            };
            
            state.framebuffer.set(x, y, {
                mod(c[0]), mod(c[1]), mod(c[2]), 255
            });
        }
    }
}


void sobel_pass(RenderState& state) {
    //sobel edge-detection
    constexpr double threshold = 0.15;
    constexpr int Gx[3][3] = { {-1,  0,  1}, {-2, 0, 2}, {-1, 0, 1} };
    constexpr int Gy[3][3] = { {-1, -2, -1}, { 0, 0, 0}, { 1, 2, 1} };

    for (int y = 1; y < state.height - 1; ++y) {
        for (int x = 1; x < state.width - 1; ++x) {
            vec2 sum;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    sum = sum + vec2{
                        Gx[j + 1][i + 1] * zbuffer[x + i + (y + j) * state.width],
                        Gy[j + 1][i + 1] * zbuffer[x + i + (y + j) * state.width]
                    };
                }
            }
            if (length(sum) > threshold)
                state.framebuffer.set(x, y, TGAColor{0, 0, 0, 255});
        }
    }
}

void geometry_pass(RenderState& state, Model& model) {
    std::cout << "ShaderType::COUNT = "
          << static_cast<int>(ShaderType::COUNT) << std::endl;

    for (int i = 0; i < static_cast<int>(ShaderType::COUNT); ++i) {

        init_zbuffer(state.width, state.height);
        init_normalBuffer(state.width, state.height);
        ShaderType shaderType = static_cast<ShaderType>(i);
        std::unique_ptr<IShader> shader;

        if (shaderType == ShaderType::TOON) {
            shader = std::make_unique<ToonShader>(model, state.light_dir, TGAColor{123, 98, 88, 255});
        }

        else if (shaderType == ShaderType::GRAY) {
            shader = std::make_unique<GrayscaleShader>(model, state.light_dir);
        }
        else {
            shader = std::make_unique<TangentTextureShader>(model, state.light_dir);
        }

        for (int f = 0; f < model.nfaces(); ++f) {// iterate through all faces
            Triangle clip = {                     // assemble primitive
                shader->vertex(f, 0),
                shader->vertex(f, 1),
                shader->vertex(f, 2)
            };
            rasterize(clip, *shader, state.framebuffer); //rasterise primitive

        }
        // ambient_occlusion_pass(state);
        // sobel_pass(state);
        std::string filename = std::string("framebuffer_") + shaderTypeToString(shaderType) + ".tga";
        std::cout << "WRITING TO FILE: " << filename << '\n';
        state.framebuffer.write_tga_file(filename.c_str());
        clear_framebuffer(state.framebuffer, {255, 255, 22, 0});
    }
}


void render_frame(RenderState& state, Model& model) {

    geometry_pass(state, model);
}

void setup_scene(const int width, const int height) {
    constexpr vec3 eye    = {-0.5, 0, 2};
    constexpr vec3 center = {0, 0, 0};
    constexpr vec3 up     = {0, 1, 0};


    lookat(eye, center, up);
    init_perspective(length(eye - center));
    init_viewport(0, 0, width, height);

}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj\n";
        return 1;
    }
    std::cout << "hihihih\n";
    constexpr int width   = 1000;
    constexpr int height  = 1000;
    constexpr vec3 light {-1, 1, 1};

    setup_scene(width, height);

    RenderState state {
        .width = width,
        .height = height,
        .framebuffer = TGAImage(width, height, TGAImage::RGB),
        .current_shader = ShaderType::GRAY,
        .light_dir = normalize(light)
    };

    for (int m=1; m<argc; m++) {
        Model model(argv[m]);
        auto start = std::chrono::high_resolution_clock::now();
        render_frame(state, model);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "CPU rendering time for " << argv[m] << ": " << elapsed.count() << " seconds\n";
    }
    
    return 0;
    
}