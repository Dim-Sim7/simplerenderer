#include <cmath>
#include <tuple>

#include "geometry.h"
#include "object_handler.h"
#include "tgaimage.h"
#include "omp.h"

constexpr int width  = 800;
constexpr int height = 800;

mat4 ModelView, Viewport, Perspective;

void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 z = normalize(eye - center);      // camera forward
    vec3 x = normalize(cross(up, z));      // camera right
    vec3 y = cross(z, x);                  // camera up

    mat4 R = {{
        { x.x, x.y, x.z, 0 },
        { y.x, y.y, y.z, 0 },
        { z.x, z.y, z.z, 0 },
        { 0,   0,   0,   1 }
    }};

    mat4 T = {{
        { 1, 0, 0, -center.x },
        { 0, 1, 0, -center.y },
        { 0, 0, 1, -center.z },
        { 0, 0, 0, 1 }
    }};

    ModelView = R * T;
}


void perspective(const double f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1/f,1}}};
}

void viewport(const int x, const int y, const int w, const int h) {
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void rasterize(const vec4 clip[3], std::vector<double> &zbuffer, TGAImage &framebuffer, const TGAColor color) {
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };                // normalized device coordinates    
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates

    mat3 ABC = {{ {screen[0].x, screen[0].y, 1.}, {screen[1].x, screen[1].y, 1.}, {screen[2].x, screen[2].y, 1.} }};

    if (ABC.det() < 1) return; // backface culling + discarding triangles that cover less than a pixel

    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners

#pragma omp parallel for //parallel processing command
    for (int x=std::max<int>(bbminx, 0); x<=std::min<int>(bbmaxx, framebuffer.width()-1); x++) { // clip the bounding box by the screen
        for (int y=std::max<int>(bbminy, 0); y<=std::min<int>(bbmaxy, framebuffer.height()-1); y++) {
            vec3 bc = ABC.invert_transpose() * vec3{static_cast<double>(x), static_cast<double>(y), 1.}; // barycentric coordinates of {x,y} w.r.t the triangle
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue; //negative barycentric coordinate => the pixel is outside the triangle

            double z = dot(bc, vec3{ ndc[0].z, ndc[1].z, ndc[2].z });
            if (z <= zbuffer[x + y * width]) continue;
            zbuffer[x + y * width] = z;
            framebuffer.set(x, y, color);
        }
    }

}


void renderModel(std::vector<vec3>& vertices, std::vector<Face>& faces, std::vector<double> &zbuffer, TGAImage &framebuffer) {

    for (const Face& f : faces) {
        vec4 clip[3];
        for (int i = 0; i < 3; i++) {
            vec3 v = vertices[f.v[i]];
            clip[i] = Perspective * ModelView * vec4{v.x, v.y, v.z, 1.0};
        }

        TGAColor rnd;
        for (int c=0; c<3; c++) rnd[c] = std::rand()%255;

        rasterize(clip, zbuffer, framebuffer, rnd);

    }
}




int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj" << std::endl;
        return 1;
    }
    constexpr vec3    eye{-1,0,2}; // camera position
    constexpr vec3 center{0,0,0};  // camera direction
    constexpr vec3     up{0,1,0};  // camera up vector

    lookat(eye, center, up);                              // build the ModelView   matrix
    perspective(length(eye-center));                        // build the Perspective matrix
    viewport(width/16, height/16, width*7/8, height*7/8); // build the Viewport    matrix

    TGAImage framebuffer(width, height, TGAImage::RGB);
    std::vector<double> zbuffer(width*height, -std::numeric_limits<double>::max());


    std::vector<vec3> vertices;
    std::vector<Face> faces;

    if (!loadObj(argv[1], vertices, faces))
        return 1;

    renderModel(vertices, faces, zbuffer, framebuffer);



    framebuffer.write_tga_file("framebuffer.tga");


    TGAImage zdebug(width, height, TGAImage::GRAYSCALE);

    double zmin =  1e9;
    double zmax = -1e9;

    for (int i = 0; i < width * height; i++) {
        double z = zbuffer[i];
        if (z == -std::numeric_limits<double>::max()) continue;
        zmin = std::min(zmin, z);
        zmax = std::max(zmax, z);
    }


    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            double z = zbuffer[x + y * width];

            if (z == -std::numeric_limits<double>::max()) {
                zdebug.set(x, y, {0}); // background
                continue;
            }

            double t = (z - zmin) / (zmax - zmin);
            unsigned char g = (unsigned char)(t * 255);
            zdebug.set(x, y, {g});
        }
    }


    zdebug.write_tga_file("zbuffer.tga");


    return 0;
}