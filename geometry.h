#pragma once
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

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
inline double dot(const vec2& a, const vec2& b) {
    return a.x * b.x + a.y * b.y;
}

inline double dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double dot(const vec4& a, const vec4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline double length(const vec2& v) {
    return std::sqrt(dot(v, v));
}

inline double length(const vec3& v) {
    return std::sqrt(dot(v, v));
}

inline double length(const vec4& v) {
    return std::sqrt(dot(v, v));
}

inline vec2 normalize(const vec2& v) {
    double len = length(v);
    assert(len > 1e-8);
    return { v.x / len, v.y / len};
}

inline vec3 normalize(const vec3& v) {
    double len = length(v);
    assert(len > 1e-8);
    return { v.x / len, v.y / len, v.z / len };
}

inline vec4 normalize(const vec4& v) {
    double len = length(v);
    assert(len > 1e-8);
    return { v.x / len, v.y / len, v.z / len, v.w / len };
}

inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// -------- Operator Overloads ----------
inline vec2 operator+(const vec2& a, const vec2& b) {
    return { a.x + b.x, a.y + b.y};
}

inline vec2 operator*(const vec2& v, double s) {
    return { v.x * s, v.y * s};
}

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

inline vec4 operator-(const vec4& a, const vec4& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

inline vec4 operator/(const vec4& v, const double s) {
    assert(std::abs(s) > 1e-8);
    return { v.x / s, v.y / s, v.z / s, v.w / s };
}

inline vec4 operator*(const vec4& v, double s) {
    return { v.x * s, v.y * s, v.z * s, v.w * s};
}

inline vec4 operator*(double s, const vec4& v) {
    return { v.x * s, v.y * s, v.z * s, v.w * s};
}

inline vec4 operator*(const vec4& a, const vec4& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

inline vec4 operator+(const vec4& v, const double& s) {
    return { v.x + s, v.y + s, v.z + s, v.w + s};
}

inline vec4 operator+(const vec4& a, const vec4& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

// -------- Matrices ----------
struct mat3 {
    double m[3][3]{};

    static mat3 identity() {
        mat3 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
        return r;
    }

    double det() const {
        return
            m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
            m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
            m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    }

    double cofactor(int r, int c) const {
        int rr[2], cc[2], ri = 0, ci = 0;
        for (int i = 0; i < 3; ++i) if (i != r) rr[ri++] = i;
        for (int j = 0; j < 3; ++j) if (j != c) cc[ci++] = j;

        double minor =
            m[rr[0]][cc[0]] * m[rr[1]][cc[1]] -
            m[rr[0]][cc[1]] * m[rr[1]][cc[0]];

        return ((r + c) & 1) ? -minor : minor;
    }

    mat3 inverse() const {
        double d = det();
        assert(std::abs(d) > 1e-8);

        mat3 inv{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                inv.m[i][j] = cofactor(j, i) / d; // NOTE swapped indices

        return inv;
    }

    mat3 transpose() const {
        mat3 t{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                t.m[i][j] = m[j][i];
        return t;
    }

    mat3 invert_transpose() const {
        // (M^{-1})^T = adj(M) / det(M)
        double d = det();
        assert(std::abs(d) > 1e-8);

        mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = cofactor(i, j) / d;

        return r;
    }


    mat3 operator/(double s) const {
        assert(std::abs(s) > 1e-8);
        mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][j] / s;
        return r;
    }

    mat3 operator*(const mat3& b) const {
        mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
        return r;
    }


    vec3 operator*(const vec3& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        };
    }


}; //END MAT3


struct mat4 {
    double m[4][4]{};

    static mat4 identity() {
        mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0;
        return r;
    }

    mat3 minor(int row, int col) const {
        mat3 r{};
        int ri = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == row) continue;
            int rj = 0;
            for (int j = 0; j < 4; ++j) {
                if (j == col) continue;
                r.m[ri][rj++] = m[i][j];
            }
            ++ri;
        }
        return r;
    }

    double cofactor(int r, int c) const {
        double d = minor(r, c).det();
        return ((r + c) & 1) ? -d : d;
    }

    double det() const {
        double d = 0.0;
        for (int c = 0; c < 4; ++c)
            d += m[0][c] * cofactor(0, c);
        return d;
    }


    mat4 invert_transpose() const {
        double d = det();
        assert(std::abs(d) > 1e-8);

        mat4 r{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = cofactor(i, j) / d;

        return r;
    }

    mat4 transpose() const {
        mat4 t{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                t.m[i][j] = m[j][i];
        return t;
    }

    // OPERATOR OVERLOADS
    mat4 operator/(double s) const {
        assert(std::abs(s) > 1e-8);
        mat4 r{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[i][j] / s;
        return r;
    }

    mat4 operator*(const mat4& B) const {
        mat4 R{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    R.m[i][j] += m[i][k] * B.m[k][j];
        return R;
    }


    vec4 operator*(const vec4& v) const {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w
        };
    }


};
