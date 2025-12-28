#pragma once
#include <cmath>
#include <cassert>
#include <iostream>

struct Face {
    int v[3];  // vertex indices
    int vt[3]; // texture coordinate indices
    int vn[3]; // normal indices
};

struct vec2 {
    double x = 0.0;
    double y = 0.0;

    double& operator[](int i) {
        assert(i >= 0 && i < 2);
        switch (i) {
            case 0: return x;
            default : return y;
        }
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 2);
        switch (i) {
            case 0: return x;
            default : return y;
        }
    }

};

struct vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double& operator[](int i) {
        assert(i >= 0 && i < 3);
        switch (i) {
            case 0: return x;
            case 1: return y;
            default : return z;
        }
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 3);
        switch (i) {
            case 0: return x;
            case 1: return y;
            default : return z;
        }
    }

};

struct vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;

    double& operator[](int i) {
        assert(i >= 0 && i < 4);
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default : return w;
        }
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 4);
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default : return w;
        }
    }

};


// -------- Vector Operations ----------
// function are inline to allow multiple definitions across .cpp files

inline double dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(const vec3& v) { // √( x2 + y2 + z2)
    return std::sqrt(dot(v, v)); // Dot(v,v) squares each component
}

inline vec3 normalize(const vec3& v) {
    double len = length(v);
    assert(len > 0);
    return { v.x / len, v.y / len, v.z / len };
}

inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// -------- Operator Overloads ----------

inline vec3 operator+(const vec3& a, const vec3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline vec3 operator-(const vec3& a, const vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline vec3 operator*(const vec3& v, double s) {
    return { v.x * s, v.y * s, v.z * s };
}

inline vec3 operator/(const vec3& v, double s) {
    assert(s != 0.0);
    return { v.x / s, v.y / s, v.z / s };
}

// -------- Matrices ----------

struct mat4 {
    double m[4][4] = {0};

    static mat4 identity() {
        mat4 r;
        for(auto i = 0; i < 4; i++) {
            r.m[i][i] = 1.0;
        }
        return r;
    }

    double* operator[](int i) {
        assert(i >= 0 && i < 4);
        return m[i];
    }

    const double* operator[](int i) const {
        assert(i >= 0 && i < 4);
        return m[i];
    }
};

// -------- Matrix x Vector ----------

inline vec4 operator*(const mat4& M, const vec4& v) {
    vec4 r;
    for (auto i = 0; i < 4; i++) {
        r[i] = M[i][0] * v.x + M[i][1] * v.y + M[i][2] * v.z + M[i][3] * v.w;
    }
    return r;
}


// -------- Matrix x Matrix ----------

inline mat4 operator*(const mat4& A, const mat4& B) {
    mat4 R;
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            for (auto k = 0; k < 4; k++) {
                R[i][j] += A[i][k] * B[k][j]; //sum of Aik * Bkj
            }
        }
    }
    return R;
}


inline vec3 transform_point(const mat4& M, const vec3& p) {
    vec4 r = M * vec4{p.x, p.y, p.z, 1.0};
    return { r.x, r.y, r.z };
}

inline vec3 transform_vector(const mat4& M, const vec3& v) {
    vec4 r = M * vec4{v.x, v.y, v.z, 0.0};
    return { r.x, r.y, r.z };
}

inline mat4 rotation_y(double a) { //transformation matrix that is used to perform a rotation in Euclidean space
    mat4 r = mat4::identity();

    r[0][0] = std::cos(a);
    r[0][2] = std::sin(a);

    r[2][0] = -std::sin(a);
    r[2][2] = std::cos(a);

    return r;
}