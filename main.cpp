#include <cstdlib>
#include <iostream>
#include "graphics_library.h"
#include "model.h"

extern mat4 ModelView, Viewport, Perspective;     // "OpenGL" state matrices and
extern std::vector<double> zbuffer;     // the depth buffer
constexpr vec3 sunRayDirection {-1, 1, 1};

struct BlankShader : IShader {
    const Model& model;

    BlankShader(const Model& m) : model(m) {}

    virtual vec4 vertex(const int face, const int vert) {
        vec4 gl_Position = ModelView * model.vert(face, vert);
        return Perspective * gl_Position;
    }

    virtual std::pair<bool,TGAColor> fragment(const vec3 bar) const {
        return {false, {255, 255, 255, 255}};
    }
};

struct PhongReflectionShader : IShader {
    const Model &model;
    vec4 lightDir;                      //light direction in eye coords

    vec2 varying_uv[3];        // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    vec4 varying_norm[3];      // normal per vertex to be interpolated by the fragment shader    
    vec4 tri[3];               // triangle in view coordinates

    PhongReflectionShader(const Model &m) : model(m) {
        lightDir = normalize((ModelView * vec4{sunRayDirection.x, sunRayDirection.y, sunRayDirection.z, 0.0})); // transform the light vector to view coordinates
    }

    //this method is called per triangle vertex (3 times for 1 face)
    virtual vec4 vertex(const int face, const int vert) {
        varying_uv[vert] = model.uv(face, vert);
        varying_norm[vert] = ModelView.invert_transpose() * model.normal(face, vert);
        vec4 gl_Position = ModelView * model.vert(face, vert);
        tri[vert] = gl_Position;
        return Perspective * gl_Position;                           // transforms to clip space (final value to project vertex onto screen)
    }

    // bar = barycentric coords of current pixel within triangle, normally used to interpolate vertex attributes
    virtual std::pair<bool, TGAColor> fragment(const vec3 bar) const {

        // --- Interpolate UVs ---
        vec2 uv =
            varying_uv[0]*bar[0] +
            varying_uv[1]*bar[1] +
            varying_uv[2]*bar[2];

        // --- Interpolate geometric normal ---
        vec4 N_geom = normalize(
            varying_norm[0]*bar[0] +
            varying_norm[1]*bar[1] +
            varying_norm[2]*bar[2]
        );

        vec4 N; // final normal used for lighting

        // --- Tangent-space construction ---
        mat2_4 E = { tri[1] - tri[0], tri[2] - tri[0] };
        mat2   U = { varying_uv[1] - varying_uv[0],
                    varying_uv[2] - varying_uv[0] };

        if (std::abs(U.det()) > 1e-8) {
            // Valid UV mapping → normal mapping enabled
            mat2_4 T = U.invert() * E;

            mat4 D = {
                normalize(T[0]),   // tangent
                normalize(T[1]),   // bitangent
                N_geom,             // normal
                {0,0,0,1}
            };

            // Tangent-space normal → view space
            N = normalize(D.transpose() * model.normal(uv));
        } else {
            // Degenerate UVs → fallback
            N = N_geom;
        }

        // --- Lighting vectors ---
        vec4 l = normalize(lightDir);
        vec4 r = normalize(N * (2.0 * dot(N, l)) - l);

        // --- Phong lighting ---
        double ambient  = 0.04;
        double diffuse  = std::max(0.0, dot(N, l));

        double shininess = 35.0;
        double specular =
            (1.0 * sample2D(model.specular(), uv)[0] / 255.0) *
            std::pow(std::max(r.z, 0.0), shininess);

        double lighting = ambient + diffuse + 0.9*specular;
        lighting = std::min(1.0, lighting);

        // --- Glow textures ---
        vec3 emissive = { 0.0, 0.0, 0.0 };
        TGAColor glow = sample2D(model.glow(), uv);
        emissive = {
        glow[0] / 255.0,
        glow[1] / 255.0,
        glow[2] / 255.0
        };

        // --- Texture fetch ---
        TGAColor gl_FragColor = sample2D(model.diffuse(), uv);

        for (int c : {0,1,2}) {
            gl_FragColor[c] = std::min<int>(255, gl_FragColor[c] * lighting + emissive[c]);
        }

        return { false, gl_FragColor };
    }

};

void drop_zbuffer(std::string filename,
                  std::vector<double> &zbuffer,
                  int width,
                  int height) {

    // Create a grayscale image to store the depth visualization
    // Each pixel will represent depth at that screen location
    TGAImage zimg(width, height, TGAImage::GRAYSCALE, {0,0,0,0});

    // Initialize min/max depth values
    // These will be used to normalize depth into [0,255]
    double minz = +1000;
    double maxz = -1000;

    // ---- First pass: find depth range ----
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {

            // Depth stored in zbuffer (one value per pixel)
            double z = zbuffer[x + y * width];

            // Ignore pixels that were never written (background)
            if (z < -100) continue;

            // Track min and max depth values actually used
            minz = std::min(z, minz);
            maxz = std::max(z, maxz);
        }
    }

    // ---- Second pass: map depth to grayscale ----
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {

            double z = zbuffer[x + y * width];
            if (z < -100) continue;

            // Normalize depth into [0,1], then scale to [0,255]
            // This makes the nearest pixel white and farthest black
            z = (z - minz) / (maxz - minz) * 255;

            // Write grayscale value into the image
            zimg.set(x, y, { z, 255, 255, 255 });
        }
    }

    // Save depth visualization to disk
    zimg.write_tga_file(filename);
}



int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj\n";
        return 1;
    }

    constexpr int width   = 1000;
    constexpr int height  = 1000;
    constexpr int shadoww = 8000;
    constexpr int shadowh = 8000;

    constexpr vec3 eye    = {-0.5, 0, 2};
    constexpr vec3 center = {0, 0, 0};
    constexpr vec3 up     = {0, 1, 0};

    vec3 light_dir = normalize(sunRayDirection);
    vec3 light_pos = center - light_dir * 10.0; // directional light approximation

    /* =========================
       CAMERA PASS
       ========================= */

    lookat(eye, center, up);
    init_perspective(length(eye - center));
    init_viewport(0, 0, width, height);
    init_zbuffer(width, height);

    TGAImage framebuffer(width, height, TGAImage::RGB, {0,0,0,255});

    for (int m = 1; m < argc; m++) {
        Model model(argv[m]);
        PhongReflectionShader shader(model);

        for (int f = 0; f < model.nfaces(); f++) {
            Triangle clip = {
                shader.vertex(f, 0),
                shader.vertex(f, 1),
                shader.vertex(f, 2)
            };
            rasterize(clip, shader, framebuffer);
        }
    }

    framebuffer.write_tga_file("framebuffer.tga");

    // Save camera depth buffer
    std::vector<double> zbuffer_camera = zbuffer;
    drop_zbuffer("zbuffer_camera.tga", zbuffer_camera, width, height);

    // Camera inverse for reconstruction
    mat4 CamInv = (Viewport * Perspective * ModelView).invert();

    /* =========================
       SHADOW MAP PASS
       ========================= */

    lookat(light_pos, center, up);

    // NOTE: directional light → orthographic would be ideal
    init_perspective(10.0); // acts as a depth range limiter
    init_viewport(0, 0, shadoww, shadowh);
    init_zbuffer(shadoww, shadowh);

    TGAImage shadow_fb(shadoww, shadowh, TGAImage::RGB, {0,0,0,255});

    for (int m = 1; m < argc; m++) {
        Model model(argv[m]);
        BlankShader shader(model);

        for (int f = 0; f < model.nfaces(); f++) {
            Triangle clip = {
                shader.vertex(f, 0),
                shader.vertex(f, 1),
                shader.vertex(f, 2)
            };
            rasterize(clip, shader, shadow_fb);
        }
    }

    shadow_fb.write_tga_file("shadowmap.tga");

    std::vector<double> zbuffer_shadow = zbuffer;
    drop_zbuffer("zbuffer_shadow.tga", zbuffer_shadow, shadoww, shadowh);

    // Light-space transform
    mat4 LightMVP = Viewport * Perspective * ModelView;

    /* =========================
       SHADOW RESOLVE PASS
       ========================= */

    std::vector<bool> lit_mask(width * height, true);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            double z = zbuffer_camera[x + y * width];
            if (z < -100) {
                lit_mask[x + y * width] = true;
                continue;
            }

            // Screen → world
            vec4 world = CamInv * vec4{x, y, z, 1.0};

            // World → light screen
            vec4 light_clip = LightMVP * world;
            vec3 light_ndc  = light_clip.xyz() / light_clip.w;

            bool lit =
                light_ndc.x < 0 || light_ndc.x >= shadoww ||
                light_ndc.y < 0 || light_ndc.y >= shadowh ||
                light_ndc.z >
                    zbuffer_shadow[int(light_ndc.x) + int(light_ndc.y) * shadoww] - 0.02;

            lit_mask[x + y * width] = lit;
        }
    }

    /* =========================
       DEBUG: SHADOW MASK
       ========================= */

    TGAImage maskimg(width, height, TGAImage::GRAYSCALE, {0,0,0,255});
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (!lit_mask[x + y * width])
                maskimg.set(x, y, {255,255,255,255});
        }
    }
    maskimg.write_tga_file("mask.tga");

    /* =========================
       APPLY SHADOWS
       ========================= */

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            if (lit_mask[x + y * width]) continue;

            TGAColor c = framebuffer.get(x, y);
            vec3 col = {c[0], c[1], c[2]};

            if (length(col) < 80) continue;

            col = normalize(col) * 80;
            framebuffer.set(x, y, {col[0], col[1], col[2], 255});
        }
    }

    framebuffer.write_tga_file("shadow.tga");

    return 0;
}
