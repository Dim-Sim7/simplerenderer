#pragma once
#include <cmath>
#include <cassert>
#include <algorithm>

//This is my templated vector math class
//Includes operator overloads and common vector functions

template <int n>
struct vec {
    float data[n] = {0};

    constexpr float& operator[](const int i) {
        assert(i >= 0 && i < n);
        return data[i];
    }

    constexpr const float& operator[](const int i) const {
        assert(i >= 0 && i < n);
        return data[i];
    }


};

template<>
struct vec<2> {
    union {
        struct { float x, y; };
        float  data[2]{};
    };

    constexpr float& operator[](int i) {
        assert(i >= 0 && i < 2);
        return data[i];
    }

    constexpr const float& operator[](int i) const {
        assert(i >= 0 && i < 2);
        return data[i];
    }
};

template<>
struct vec<3> {
    union {
        struct { float x, y, z; };
        float data[3];
    };

    constexpr float& operator[](int i) {
        assert(i >= 0 && i < 3);
        return data[i];
    }

    constexpr const float& operator[](int i) const {
        assert(i >= 0 && i < 3);
        return data[i];
    }

    [[nodiscard]] constexpr vec<2> xy() const { return {x, y}; }
    [[nodiscard]] constexpr vec<2> xz() const { return {x, z}; }
    [[nodiscard]] constexpr vec<2> yz() const { return {y, z}; }
};

template<>
struct vec<4> {
    union {
        struct { float x, y, z, w; };
        float data[4];
    };

    constexpr float& operator[](int i) {
        assert(i >= 0 && i < 4);
        return data[i];
    }

    constexpr const float& operator[](int i) const {
        assert(i >= 0 && i < 4);
        return data[i];
    }
    [[nodiscard]] constexpr vec<2> xy() const { return {x, y}; }
    [[nodiscard]] constexpr vec<2> xz() const { return {x, z}; }
    [[nodiscard]] constexpr vec<2> yz() const { return {y, z}; }
    [[nodiscard]] constexpr vec<3> xyz() const { return {x, y, z}; }
};



using vec2 = vec<2>;
using vec3 = vec<3>;
using vec4 = vec<4>;

// -------- Operator Overloads ----------

//VECTOR WITH VECTOR
template<int n> 
constexpr vec<n> operator+(const vec<n>& a, const vec<n>& b) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<int n> 
constexpr vec<n> operator-(const vec<n>& a, const vec<n>& b) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template<int n> 
constexpr vec<n> operator*(const vec<n>& a, const vec<n>& b) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

template<int n> 
constexpr vec<n> operator/(const vec<n>& a, const vec<n>& b) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        assert(b[i] != 0);
        result[i] = a[i] / b[i];
    }
    return result;
}


//VECTOR WITH SCALAR
template<int n> 
constexpr vec<n> operator+(const vec<n>& a, float s) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] + s;
    }
    return result;
}

template<int n> 
constexpr vec<n> operator-(const vec<n>& a, float s) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] - s;
    }
    return result;
}

template<int n> 
constexpr vec<n> operator*(const vec<n>& a, float s) {
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] * s;
    }
    return result;
}

template<int n>
constexpr vec<n> operator*(float s, const vec<n>& v) {
    return v * s;
}

template<int n> 
constexpr vec<n> operator/(const vec<n>& a, float s) {
    assert(s != 0);
    vec<n> result;
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] / s;
    }
    return result;
}

// -------- Vector Operations ----------

template<int n>
constexpr float dot(const vec<n>& a, const vec<n>& b) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

template<int n>
float length(const vec<n>& v) {
    return std::sqrt(dot(v, v));
}

template<int n>
vec<n> normalize(const vec<n>& v) {
    float len = length(v);
    vec<n> result;
    assert(len > 1e-8);
    for (int i = 0; i < n; ++i)
        result[i] = v[i] / len;
    return result;
}

template<int n>
constexpr vec<n> cross(const vec<n>& a, const vec<n>& b) {
    static_assert(n == 3, "cross() is only defined for 3D vectors");
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}


