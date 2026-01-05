#include <vector>
#include "geometry.h"
#include "tgaimage.h"
struct Face {
    int v[3];  // vertex indices
    int vt[3]; // texture coordinate indices
    int vn[3]; // normal indices
};
class Model {
    std::vector<vec4> vertices = {};    // array of vertices
    std::vector<vec4> normals = {};     // normals of vertices
    std::vector<vec2> tex = {};         // texture coords u,v
    std::vector<Face> faces = {}; // per-triangle index in the above array

    TGAImage normalmap = {};       // normal map texture
    TGAImage specularmap = {};
    TGAImage glowmap = {};          
    TGAImage diffusemap = {};       // colour map
public:
    Model(const std::string& filename);
    int nverts() const; // number of vertices
    int nfaces() const; // number of triangles
    vec4 vert(const int i) const;                          // 0 <= i < nverts()
    vec4 vert(const int iface, const int nthvert) const;   // 0 <= iface < nfaces(), 0 <= nthvert < 3
    
    vec4 normal(const int iface, const int nthvert) const;
    vec4 normal(const vec2& uv) const;
    double specular(const vec2& uv) const;
    double glow(const vec2& uv) const;
    TGAColor diffuse(const vec2& uv) const;
    vec2 uv(const int iface, const int nthvert) const;

    const TGAImage& diffuse() const;
    const TGAImage& specular() const;
    const TGAImage& glow() const;
};
