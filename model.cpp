

#include <fstream>
#include <sstream>
#include <iostream>
#include "model.h"

Model::Model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open OBJ file: " << filename << "\n";
        return;                              // constructor finishes – model stays empty
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream ss(line);
        std::string type;
        ss >> type;

        // -------------------- Vertex position --------------------
        if (type == "v") {
            vec4 v = {0, 0, 0, 1};
            if (ss >> v.x >> v.y >> v.z) {
                vertices.push_back(v);
            } else {
                std::cerr << "Warning: malformed vertex line: " << line << "\n";
            }
        }
        // -------------------- Texture --------------------
        else if (type == "vt") {
            vec2 t;
            if (ss >> t.x >> t.y) {
                tex.push_back({t.x, 1-t.y}); 
            } else {
                std::cerr << "Warning: malformed normal line: " << line << "\n";
            }
        }
        // -------------------- Vertex normal --------------------
        else if (type == "vn") {
            vec4 n;
            if (ss >> n.x >> n.y >> n.z) {
                normals.push_back(normalize(n)); // normalize once on load
            } else {
                std::cerr << "Warning: malformed normal line: " << line << "\n";
            }
        }
        // -------------------- Face --------------------
        else if (type == "f") {
            Face f{{0,0,0}, {0,0,0}, {0,0,0}};   // zero-init (important!)

            std::string token;
            for (int i = 0; i < 3; ++i) {
                if (!(ss >> token)) {
                    std::cerr << "Warning: face has fewer than 3 vertices: " << line << "\n";
                    break;
                }

                // Parse the triple v/vt/vn (any of the three may be missing)
                int v_idx = 0, vt_idx = 0, vn_idx = 0;

                size_t pos1 = token.find('/');
                size_t pos2 = token.find('/', pos1 + 1);

                // v
                if (pos1 != std::string::npos)
                    v_idx = std::stoi(token.substr(0, pos1));
                else
                    v_idx = std::stoi(token);

                // vt (if present)
                if (pos1 != std::string::npos && pos2 != std::string::npos && pos2 > pos1 + 1)
                    vt_idx = std::stoi(token.substr(pos1 + 1, pos2 - pos1 - 1));
                else if (pos1 != std::string::npos && pos2 == std::string::npos)
                    vt_idx = std::stoi(token.substr(pos1 + 1));

                // vn (if present)
                if (pos2 != std::string::npos)
                    vn_idx = std::stoi(token.substr(pos2 + 1));

                // OBJ indices are 1-based → convert to 0-based
                f.v[i]  = v_idx  - 1;
                f.vt[i] = vt_idx - 1;
                f.vn[i] = vn_idx - 1;
            }
            faces.push_back(f);
        }
        
    }

    std::cout << "# v# " << nverts() << " f# " << nfaces() << "\n";

    auto load_texture = [&filename](const std::string suffix, TGAImage &img) {
        size_t dot = filename.find_last_of(".");
        if (dot==std::string::npos) return;
        std::string texfile = filename.substr(0,dot) + suffix;
        std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
    };
    load_texture("_nm_tangent.tga", normalmap);
    load_texture("_spec.tga", specularmap);
    load_texture("_glow.tga", glowmap);
    load_texture("_diffuse.tga", diffusemap);
}

int Model::nverts() const {
    return static_cast<int>(vertices.size());
}

int Model::nfaces() const {
    return static_cast<int>(faces.size());       // each Face is one triangle
}

vec4 Model::vert(const int i) const {
    return vertices[i];
}

vec4 Model::vert(const int iface, const int nthvert) const {
    return vertices[faces[iface].v[nthvert]];
}

vec4 Model::normal(const int iface, const int nthvert) const {
    int ni = faces[iface].vn[nthvert];
    assert(ni >= 0 && ni < normals.size());
    return normals[ni];
}

vec4 Model::normal(const vec2& uv) const {
    int x = std::min(
        normalmap.width() - 1,
        std::max(0, int(uv.x * normalmap.width()))
    );

    int y = std::min(
        normalmap.height() - 1,
        std::max(0, int(uv.y * normalmap.height()))
    );

    TGAColor c = normalmap.get(x, y);
 
    vec4 n = vec4{(double)c[2],(double)c[1],(double)c[0],0} * 2.0 / 255.0 - vec4{1,1,1,0};

    return normalize(n);
}

double Model::specular(const vec2& uv) const {
    int x = std::min(
    specularmap.width()  - 1,
    std::max(0, int(uv.x * specularmap.width()))
    );

    int y = std::min(
        specularmap.height() - 1,
        std::max(0, int(uv.y * specularmap.height()))
    );

    TGAColor c = specularmap.get(x, y);

    return c[0] / 255.0;
}

double Model::glow(const vec2& uv) const {
    int x = uv.x * glowmap.width();
    int y = (1 - uv.y) * glowmap.height();
    return glowmap.get(x, y)[0] / 255.0; // usually grayscale
}


TGAColor Model::diffuse(const vec2& uv) const {
    int x = std::min(
        diffusemap.width() - 1,
        std::max(0, int(uv.x * diffusemap.width()))
    );

    int y = std::min(
        diffusemap.height() - 1,
        std::max(0, int(uv.y * diffusemap.height()))
    );

    return diffusemap.get(x, y);
}

vec2 Model::uv(const int iface, const int nthvert) const {
    int ti = faces[iface].vt[nthvert];
    assert(ti >= 0 && ti < tex.size());
    return tex[ti];
}


const TGAImage& Model::diffuse() const {
    return diffusemap;
}
const TGAImage& Model::specular() const {
    return specularmap;
}
const TGAImage& Model::glow() const {
    return glowmap;
}