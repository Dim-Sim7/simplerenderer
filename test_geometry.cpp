// test_math.cpp
#include <cassert>
#include <cmath>
#include <iostream>

#include "geometry.h"

// ---------- Helpers ----------
constexpr double EPS = 1e-6;

bool feq(double a, double b, double eps = EPS) {
    return std::abs(a - b) < eps;
}

bool feq_vec2(const vec2& a, const vec2& b) {
    return feq(a.x, b.x) && feq(a.y, b.y);
}

bool feq_vec3(const vec3& a, const vec3& b) {
    return feq(a.x, b.x) && feq(a.y, b.y) && feq(a.z, b.z);
}

bool feq_vec4(const vec4& a, const vec4& b) {
    return feq(a.x, b.x) && feq(a.y, b.y) && feq(a.z, b.z) && feq(a.w, b.w);
}

// ---------- vec2 ----------
void test_vec2() {
    vec2 a{3, 4};
    vec2 b{1, 2};

    assert(a[0] == 3 && a[1] == 4);
    assert(feq(dot(a, b), 3 * 1 + 4 * 2));
    assert(feq(length(a), 5));

    vec2 n = normalize(a);
    assert(feq(length(n), 1.0));

    vec2 c = a + b;
    assert(feq_vec2(c, {4, 6}));

    vec2 d = a * 2.0;
    assert(feq_vec2(d, {6, 8}));
}

// ---------- vec3 ----------
void test_vec3() {
    vec3 a{1, 0, 0};
    vec3 b{0, 1, 0};

    assert(feq(dot(a, b), 0.0));

    vec3 c = cross(a, b);
    assert(feq_vec3(c, {0, 0, 1}));

    vec3 d = a + b;
    assert(feq_vec3(d, {1, 1, 0}));

    vec3 e = d - a;
    assert(feq_vec3(e, b));

    vec3 f = d * 2.0;
    assert(feq_vec3(f, {2, 2, 0}));

    vec3 g = f / 2.0;
    assert(feq_vec3(g, d));

    vec3 n = normalize(vec3{0, 3, 4});
    assert(feq(length(n), 1.0));
}

// ---------- vec4 ----------
void test_vec4() {
    vec4 a{1, 2, 3, 1};
    vec4 b{2, 3, 4, 1};

    vec4 c = a + b;
    assert(feq_vec4(c, {3, 5, 7, 2}));

    vec4 d = a * 2.0;
    assert(feq_vec4(d, {2, 4, 6, 2}));

    vec4 e = d / 2.0;
    assert(feq_vec4(e, a));

    assert(feq(dot(a, b), 1*2 + 2*3 + 3*4 + 1*1));
}

// ---------- mat3 ----------
void test_mat3() {
    mat3 I = mat3::identity();

    vec3 v{1, 2, 3};
    vec3 r = I * v;
    assert(feq_vec3(r, v));

    mat3 M{};
    M.m[0][0] = 1; M.m[0][1] = 2; M.m[0][2] = 3;
    M.m[1][0] = 0; M.m[1][1] = 1; M.m[1][2] = 4;
    M.m[2][0] = 5; M.m[2][1] = 6; M.m[2][2] = 0;

    double det = M.det();
    assert(feq(det, 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)));

    mat3 inv = M.inverse();
    mat3 shouldBeI = M * inv;

    mat3 b = M * M.inverse();
    vec3 a = b * vec3{1,2,3};
    assert(feq_vec3(a, {1,2,3}));

    vec3 test = shouldBeI * vec3{1, 0, 0};
    std::cout << test.x << " " << test.y << " " << test.z << '\n';
    assert(feq_vec3(test, {1, 0, 0}));

    assert(feq_vec3((M * M.inverse()) * vec3{1,2,3}, {1,2,3}));
    assert(feq_vec3((M.inverse() * M) * vec3{4,5,6}, {4,5,6}));
}

// ---------- mat4 ----------
void test_mat4() {
    mat4 I = mat4::identity();

    vec4 v{1, 2, 3, 1};
    vec4 r = I * v;
    assert(feq_vec4(r, v));

    mat4 T = mat4::identity();
    T.m[0][3] = 5;
    T.m[1][3] = -3;
    T.m[2][3] = 2;

    vec4 vt = T * v;
    assert(feq_vec4(vt, {6, -1, 5, 1}));

    mat4 TT = T.transpose();
    assert(TT.m[3][0] == 5);
    assert(TT.m[3][1] == -3);
    assert(TT.m[3][2] == 2);
}

// ---------- Entry ----------
int main() {
    test_vec2();
    test_vec3();
    test_vec4();
    test_mat3();
    test_mat4();

    std::cout << "All math unit tests passed ✅\n";
    return 0;
}