#include "tangent_texture_shader.h"

TangentTextureShader::TangentTextureShader(const Model& model, const vec3& light) 
        : model(model) {
    l = normalize((ModelView*vec4{light.x, light.y, light.z, 0.}).xyz()); // transform the light vector to view coordinates
}

vec4 TangentTextureShader::vertex(const int face, const int vert) {
    varying_uv[vert] = model.uv(face, vert);
    //Transform normal to view space, w = 0 because this is a direction, not a position
    varying_norm[vert] = (ModelView.invert_transpose() * model.normal(face, vert)).xyz();

    vec4 view_pos = ModelView * model.vert(face, vert); //transform vertex to view space

    //store view space for TBN construction
    varying_pos[vert] = view_pos.xyz();
    return Perspective * view_pos; //Project to clip space
}

Fragment TangentTextureShader::fragment(const vec3 bar) const {
    //Interpolate view space normal (normalize because it is a direction, not a location)
    vec3 Ng = normalize(
        varying_norm[0] * bar[0] +
        varying_norm[1] * bar[1] +
        varying_norm[2] * bar[2]
    );

    //Triangle edges in view space
    vec3 e1 = varying_pos[1] - varying_pos[0];
    vec3 e2 = varying_pos[2] - varying_pos[0];

    //Triangle edges in UV space
    vec2 duv1 = varying_uv[1] - varying_uv[0];
    vec2 duv2 = varying_uv[2] - varying_uv[0];

    //Build tangent + bitangent (view space)
    double det = duv1.x * duv2.y - duv1.y * duv2.x;

    double invDet = 1.0 / det;
    vec3 T = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
    vec3 B = normalize((-1.0*e1 * duv2.x + e2 * duv1.x) * invDet);

    //TBN matrix
    if (dot(cross(T, B), Ng) < 0.0)
        T = T * -1.0;
    mat3 TBN = mat3{ T, B, Ng };

    vec2 uv = varying_uv[0] * bar[0] + varying_uv[1] * bar[1] + varying_uv[2] * bar[2]; //interpolate UV coords for this pixel

    //Sample tangent space normal
    TGAColor c = sample2D(model.normal(), uv);
    vec3 nm = vec3{
        c[2] / 255.0,
        c[1] / 255.0,
        c[0] / 255.0
    };
    nm = nm * 2.0 - vec3{1,1,1};
    nm = normalize(nm);
    nm.y = nm.y * -1.0;
    vec3 Nv = normalize(TBN * normalize(nm)); //micro-surface detail of model
    
    //Convert to view space
    
    //lighting bling-phong
    double ambient = 0.4;
    double diffuse = std::max(0.0, dot(Nv, l));

    vec3 r = normalize(Nv * (2.0 * dot(Nv, l)) - l);
    double specular = std::pow(std::max(dot(r, vec3{0,0,1}), 0.0), 35);
    //std::cout << ambient + diffuse + specular << '\n';
    TGAColor gl_FragColor = sample2D(model.diffuse(), uv);
    for (int channel : {0, 1, 2}) {
        gl_FragColor[channel] = std::min<int>(255, gl_FragColor[channel]*(ambient + diffuse + specular));
    }
    return Fragment{false, gl_FragColor, Nv};   
}

