#pragma once
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

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
        return i == 0 ? x : y;
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 2);
        return i == 0 ? x : y;
    }
};

struct vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double& operator[](int i) {
        assert(i >= 0 && i < 3);
        return i == 0 ? x : (i == 1 ? y : z);
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 3);
        return i == 0 ? x : (i == 1 ? y : z);
    }
};

struct vec4 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 0.0;

    double& operator[](int i) {
        assert(i >= 0 && i < 4);
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }

    const double& operator[](int i) const {
        assert(i >= 0 && i < 4);
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }

    vec2 xy() const { return {x, y}; }
    vec2 xz() const { return {x, z}; }
    vec2 yz() const { return {y, z}; }
    vec3 xyz() const { return {x, y, z}; }
};

// -------- Vector Operations ----------
inline double dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(const vec3& v) {
    return std::sqrt(dot(v, v));
}

inline vec3 normalize(const vec3& v) {
    double len = length(v);
    assert(len > 1e-8);
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

inline vec3 operator*(double s, const vec3& v) { return v * s; }

inline vec3 operator*(const vec3& a, const vec3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline vec3 operator/(const vec3& v, double s) {
    assert(std::abs(s) > 1e-8);
    return { v.x / s, v.y / s, v.z / s };
}

inline vec4 operator/(const vec4& v, const double s) {
    assert(std::abs(s) > 1e-8);
    return { v.x / s, v.y / s, v.z / s, v.w / s };
}

// -------- Matrices ----------
struct mat3 {
    double m[3][3] = {}; // zero-initialized

    static mat3 identity() {
        mat3 r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
        return r;
    }

    double det() const {
        return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
               m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
               m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    }

    // Cleaner inverse using cofactors
    mat3 inverse() const {
        double d = det();
        assert(std::abs(d) >= 1e-8 && "Singular matrix inversion");

        mat3 adjugate;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                adjugate.m[j][i] = cofactor(i, j);  // transpose during assignment

        return adjugate / d;
    }

    mat3 transpose() const {
        mat3 t{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                t.m[i][j] = m[j][i];
        return t;
    }

    double cofactor(int row, int col) const {
        int rows[2], cols[2];
        int ri = 0, ci = 0;
        for (int r = 0; r < 3; ++r) if (r != row) rows[ri++] = r;
        for (int c = 0; c < 3; ++c) if (c != col) cols[ci++] = c;

        double minor = m[rows[0]][cols[0]] * m[rows[1]][cols[1]] -
                       m[rows[0]][cols[1]] * m[rows[1]][cols[0]];
        return ((row + col) % 2 == 0 ? minor : -minor);
    }

    mat3 invert_transpose() const {
        mat3 adjugate_transpose{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                adjugate_transpose.m[i][j] = cofactor(i, j);

        double d = det();
        assert(std::abs(d) >= 1e-8);
        return adjugate_transpose / d;
    }

    double* operator[](int i) {
        assert(i >= 0 && i < 3);
        return m[i];
    }

    const double* operator[](int i) const {
        assert(i >= 0 && i < 3);
        return m[i];
    }

    mat3 operator/(double d) const {
        assert(std::abs(d) > 1e-8);
        mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] / d;
        return r;
    }
};

struct mat4 {
    double m[4][4] = {}; // zero-initialized

    static mat4 identity() {
        mat4 r{};
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0;
        return r;
    }

    mat4 transpose() const {
        mat4 t{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                t.m[i][j] = m[j][i];
        return t;
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
inline vec3 operator*(const mat3& M, const vec3& v) {
    return {
        M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z,
        M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z,
        M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z
    };
}

inline vec4 operator*(const mat4& M, const vec4& v) {
    return {
        M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z + M.m[0][3]*v.w,
        M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z + M.m[1][3]*v.w,
        M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z + M.m[2][3]*v.w,
        M.m[3][0]*v.x + M.m[3][1]*v.y + M.m[3][2]*v.z + M.m[3][3]*v.w
    };
}

// -------- Matrix x Matrix ----------
inline mat3 operator*(const mat3& A, const mat3& B) {
    mat3 R{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                R.m[i][j] += A.m[i][k] * B.m[k][j];
    return R;
}

inline mat4 operator*(const mat4& A, const mat4& B) {
    mat4 R{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                R.m[i][j] += A.m[i][k] * B.m[k][j];
    return R;
}

// -------- Transform Helpers ----------
inline vec3 transform_point(const mat4& M, const vec3& p) {
    vec4 r = M * vec4{p.x, p.y, p.z, 1.0};
    return {r.x, r.y, r.z};
}

inline vec3 transform_vector(const mat4& M, const vec3& v) {
    vec4 r = M * vec4{v.x, v.y, v.z, 0.0};
    return {r.x, r.y, r.z};
}

inline mat4 rotation_y(double a) {
    mat4 r = mat4::identity();
    double c = std::cos(a);
    double s = std::sin(a);
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}