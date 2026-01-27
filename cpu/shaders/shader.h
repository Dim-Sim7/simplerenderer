#pragma once
#include <utility>
#include "../graphics_library/tgaimage.h"
#include "../math/geometry.h"
#include "../graphics_library/model.h"


extern mat4 ModelView, Viewport, Perspective; 
enum ShaderType {TOON, GRAY, TEXTURE, COUNT};

struct Fragment {
    bool discard;
    TGAColor color;
    vec3 normal;
};

class IShader {
public:
    static TGAColor sample2D(const TGAImage &img, const vec2 &uvf) {
        return img.get(uvf[0] * img.width(), uvf[1] * img.height());
    }
    virtual vec4 vertex(int face, int vert) = 0;
    virtual Fragment fragment(const vec3 bar) const = 0;
    virtual ~IShader() = default;
};