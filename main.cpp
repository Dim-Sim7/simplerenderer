#include <cmath>
#include <tuple>

#include "geometry.h"
#include "object_handler.h"
#include "tgaimage.h"

constexpr int width  = 800;
constexpr int height = 800;



double signed_triangle_area(int ax, int ay, int bx, int by, int cx, int cy) {
    return .5*((by-ay)*(bx+ax) + (cy-by)*(cx+bx) + (ay-cy)*(ax+cx));
}

void triangle(
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    std::vector<double> &zbuffer, TGAImage &framebuffer, TGAColor color) {

    int bbminx = std::min(std::min(ax, bx), cx); // bounding box for the triangle
    int bbminy = std::min(std::min(ay, by), cy); // defined by top left and bottom right corners
    int bbmaxx = std::max(std::max(ax, bx), cx);
    int bbmaxy = std::max(std::max(ay, by), cy);

    bbminx = std::max(0, bbminx);
    bbminy = std::max(0, bbminy);
    bbmaxx = std::min(width  - 1, bbmaxx);
    bbmaxy = std::min(height - 1, bbmaxy);
    
    if (bbminx > bbmaxx || bbminy > bbmaxy) return;

    double total_area = signed_triangle_area(ax, ay, bx, by, cx, cy);
    if (std::abs(total_area) < 1e-6) return; //backface culling + discarding triangles that cover less than a pixel

    //check if a pixel does not belong to triangle using barycentric coords
    // iterate through box, each pixel, compute barycentric coords. if at least 1 negative component, it is outside triangle

#pragma omp parallel for //paralle processing command
    for (int x=bbminx; x<=bbmaxx; x++) {
        for (int y=bbminy; y<=bbmaxy; y++) {
            double alpha = signed_triangle_area(x, y, bx, by, cx, cy) / total_area;
            double beta  = signed_triangle_area(x, y, cx, cy, ax, ay) / total_area;
            double gamma = signed_triangle_area(x, y, ax, ay, bx, by) / total_area;

            if (alpha < 0 || beta < 0 || gamma < 0) continue; //negative barycentric coordinate => the pixel is outside the triangle
            double z = alpha * az + beta * bz + gamma * cz;
            if (z <= zbuffer[x+y*width]) continue;
            zbuffer[x+y*width] = z;
            framebuffer.set(x, y, color);
        }
    }

}



vec3 rot(vec3 v) {
    constexpr double a = M_PI/6;
    constexpr mat<3,3> Ry = {{{std::cos(a), 0, std::sin(a)}, {0,1,0}, {-std::sin(a), 0, std::cos(a)}}};
    return Ry*v;
}

std::tuple<int,int,int> project(vec3 v) { // First of all, (x,y) is an orthogonal projection of the vector (x,y,z).
    return { (v.x + 1.) *  width/2,       // Second, since the input models are scaled to have fit in the [-1,1]^3 world coordinates,
             (v.y + 1.) * height/2,       // we want to shift the vector (x,y) and then scale it to span the entire screen.
             (v.z + 1.) *   255./2 };
}


vec3 persp(vec3 v) {
    constexpr double c = 3.;
    return v / (1-v.z/c);
}

void renderModel(std::vector<vec3>& vertices, std::vector<Face>& faces, std::vector<double> &zbuffer, TGAImage &framebuffer, int width, int height) {

    for (const Face& f : faces) {

        vec3 v1 = vertices[f.v[0]]; //a
        vec3 v2 = vertices[f.v[1]]; //b
        vec3 v3 = vertices[f.v[2]]; //c

        //normalise coords into the render space
        auto [x1, y1, z1] = project(persp(rot(v1)));
        auto [x2, y2, z2] = project(persp(rot(v2)));
        auto [x3, y3, z3] = project(persp(rot(v3)));

        TGAColor rnd;
        //render triangles
        for (int c=0; c<3; c++) rnd[c] = std::rand()%255;
            triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, zbuffer, framebuffer, rnd);

    }
}




int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.obj" << std::endl;
        return 1;
    }
    TGAImage framebuffer(width, height, TGAImage::RGB);
    std::vector<double> zbuffer(width*height, -std::numeric_limits<double>::max());


    std::vector<vec3> vertices;
    std::vector<Face> faces;

    if (!loadObj(argv[1], vertices, faces))
        return 1;

    renderModel(vertices, faces, zbuffer, framebuffer, width, height);



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
