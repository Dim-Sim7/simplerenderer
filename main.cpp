#include <cstdlib>
#include <iostream>
#include "graphics_library.h"
#include "model.h"

extern mat4 ModelView, Perspective;     // "OpenGL" state matrices and
extern std::vector<double> zbuffer;     // the depth buffer
constexpr vec3 sunRayDirection {1, -1, 1};

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
    double ambient  = 0.4;
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


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj" << std::endl;
        return 1;
    }
    constexpr int width  = 1000;
    constexpr int height = 1000;

    constexpr vec3    eye{-0.5,0,2}; // camera position
    constexpr vec3 center{0,0,0};  // camera direction
    constexpr vec3     up{0,1,0};  // camera up vector


    lookat(eye, center, up);                                     // build the ModelView   matrix
    init_perspective(length(eye-center));                        // build the Perspective matrix
    init_viewport(width/16, height/16, width*7/8, height*7/8);   // build the Viewport    matrix
    init_zbuffer(width, height); 
    TGAImage framebuffer(width, height, TGAImage::RGB);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            framebuffer.set(x, y, TGAColor{0, 0, 0});
        }
    }

    for (int m = 1; m < argc; m++) {                //iterate through all inputs
        Model model(argv[m]);                       // load data into vertices and faces
        PhongReflectionShader shader(model);
        for (int f = 0; f < model.nfaces(); f++) {  //iterate through all faces
            Triangle clip = { shader.vertex(f, 0), //assemble primitive
                              shader.vertex(f, 1),
                              shader.vertex(f, 2) };
            rasterize(clip, shader, framebuffer);  //rasterize the primitive
            
        }

    }
    

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}