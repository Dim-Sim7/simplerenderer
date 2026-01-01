#include <cstdlib>
#include "graphics_library.h"
#include "model.h"

extern mat4 ModelView, Perspective;     // "OpenGL" state matrices and
extern std::vector<double> zbuffer;     // the depth buffer
constexpr vec3 sunRayDirection {1, -1, 1};

struct PhongReflectionShader : IShader {
    const Model &model;
    vec4 lightDir;                      //light direction in eye coords

    vec2 varying_uv[3];              //vn -- normal per vertex to be interp by frag shader
    vec4 varying_pos[3];            

    PhongReflectionShader(const Model &m) : model(m) {
        lightDir = normalize((ModelView * vec4{sunRayDirection.x, sunRayDirection.y, sunRayDirection.z, 0.0})); // transform the light vector to view coordinates
    }

    //this method is called per triangle vertex (3 times for 1 face)
    virtual vec4 vertex(const int face, const int vert) {
        varying_uv[vert] = model.uv(face, vert);
        vec4 gl_Position = ModelView * model.vert(face, vert);
        varying_pos[vert] = gl_Position;
        return Perspective * gl_Position;                           // transforms to clip space (final value to project vertex onto screen)
    }

    // bar = barycentric coords of current pixel within triangle, normally used to interpolate vertex attributes
    virtual std::pair<bool, TGAColor> fragment(const vec3 bar) const {
        TGAColor gl_FragColor = { 255, 255, 255, 255 };     // output color of the fragment
                                       //constant fill light
        double diffuse;
        double specular;
        double ambient = 0.2;

        vec2 uv = varying_uv[0] * bar[0] + varying_uv[1] * bar[1] + varying_uv[2] * bar[2];
        vec4 normal = normalize(ModelView.invert_transpose() * model.normal(uv));
        vec4 reflection = normalize(normal * (normal * lightDir) * 2.0  - lightDir);
        
        diffuse = std::max(0.0, dot(normal, lightDir));

        vec4 fragPos =
            varying_pos[0] * bar[0] +
            varying_pos[1] * bar[1] +
            varying_pos[2] * bar[2];

        vec4 viewDir = normalize(-1.0 * fragPos);

        double specMap = model.specular(uv);
        double shininess = 35;

        specular = specMap * std::pow(
            std::max(dot(reflection, viewDir), 0.0),
            shininess
        );

        for (int channel : {0, 1, 2})   
            gl_FragColor[channel] *= std::min(1.0, ambient + 0.4*diffuse + 0.9*specular);

        return {false, gl_FragColor};                      //final color to write to framebuffer
    }
};


int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj" << std::endl;
        return 1;
    }
    constexpr int width  = 1000;
    constexpr int height = 1000;

    constexpr vec3    eye{-1,0,2}; // camera position
    constexpr vec3 center{0,0,0};  // camera direction
    constexpr vec3     up{0,1,0};  // camera up vector


    lookat(eye, center, up);                                     // build the ModelView   matrix
    init_perspective(length(eye-center));                        // build the Perspective matrix
    init_viewport(width/16, height/16, width*7/8, height*7/8);   // build the Viewport    matrix
    init_zbuffer(width, height); 

    
    

    TGAImage framebuffer(width, height, TGAImage::RGB);

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