// test_geometry.cpp
#include <cassert>
#include <cmath>
#include <iostream>

#include "vector.h"
#include "matrix.h"

// --------------------------------------------------
// Helpers
// --------------------------------------------------

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
    return feq(a.x, b.x) && feq(a.y, b.y) &&
           feq(a.z, b.z) && feq(a.w, b.w);
}

// --------------------------------------------------
// vec2
// --------------------------------------------------

void test_vec2() {
    vec2 a{3, 4};
    vec2 b{1, 2};

    assert(a[0] == 3 && a[1] == 4);
    assert(feq(dot(a, b), 11));
    assert(feq(length(a), 5));

    vec2 n = normalize(a);
    assert(feq(length(n), 1.0));

    assert(feq_vec2(a + b, {4, 6}));
    assert(feq_vec2(a * 2.0, {6, 8}));
}

// --------------------------------------------------
// vec3
// --------------------------------------------------

void test_vec3() {
    vec3 a{1, 0, 0};
    vec3 b{0, 1, 0};

    assert(feq(dot(a, b), 0.0));
    assert(feq_vec3(cross(a, b), {0, 0, 1}));

    vec3 c = a + b;
    assert(feq_vec3(c, {1, 1, 0}));

    vec3 n = normalize(vec3{0, 3, 4});
    assert(feq(length(n), 1.0));
}

// --------------------------------------------------
// vec4
// --------------------------------------------------

void test_vec4() {
    vec4 a{1, 2, 3, 1};
    vec4 b{2, 3, 4, 1};

    assert(feq_vec4(a + b, {3, 5, 7, 2}));
    assert(feq_vec4(a * 2.0, {2, 4, 6, 2}));
    assert(feq(dot(a, b), 1*2 + 2*3 + 3*4 + 1*1));
}

// --------------------------------------------------
// mat3 basics
// --------------------------------------------------

void test_mat3_identity() {
    mat3 I = mat3::identity();
    vec3 v{1, 2, 3};

    assert(feq_vec3(I * v, v));
}

void test_mat3_determinant() {
    mat3 I = mat3::identity();
    assert(feq(I.det(), 1.0));

    mat3 Z{};
    assert(feq(Z.det(), 0.0));

    mat3 S{};
    S[0][0] = 2;
    S[1][1] = 3;
    S[2][2] = 4;
    assert(feq(S.det(), 24.0));
}

void test_mat3_inverse() {
    mat3 M{};
    M[0] = {1, 2, 3};
    M[1] = {0, 1, 4};
    M[2] = {5, 6, 0};

    mat3 inv = M.invert();
    mat3 I = M * inv;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(feq(I[i][j], (i == j ? 1.0 : 0.0)));
}

// --------------------------------------------------
// Normal matrix (inverse-transpose)
// --------------------------------------------------

void test_normal_matrix() {
    mat3 M{};
    M[0][0] = 2;
    M[1][1] = 3;
    M[2][2] = 4;

    mat3 N = M.invert_transpose();

    vec3 n{0, 0, 1};
    vec3 tn = normalize(N * n);

    assert(feq_vec3(tn, {0, 0, 1}));
}

// --------------------------------------------------
// mat4 (homogeneous transforms)
// --------------------------------------------------

void test_mat4_identity() {
    mat4 I = mat4::identity();
    vec4 v{1, 2, 3, 1};

    assert(feq_vec4(I * v, v));
}

void test_mat4_translation() {
    mat4 T = mat4::identity();
    T[0][3] = 5;
    T[1][3] = -3;
    T[2][3] = 2;

    vec4 p{1, 2, 3, 1};
    vec4 r = T * p;

    assert(feq_vec4(r, {6, -1, 5, 1}));
}

void test_mat4_translation_composition() {
    mat4 Tx = mat4::identity();
    Tx[0][3] = 3;

    mat4 Ty = mat4::identity();
    Ty[1][3] = 4;

    mat4 T = Ty * Tx;

    vec4 p{1, 2, 3, 1};
    vec4 r = T * p;

    assert(feq_vec4(r, {4, 6, 3, 1}));
}

// --------------------------------------------------
// Affine / screen-space style test
// --------------------------------------------------

void test_affine_2d_transform() {
    mat3 A = mat3::identity();
    A[0][2] = 10;
    A[1][2] = 20;

    vec3 p{1, 2, 1};
    vec3 r = A * p;

    assert(feq_vec3(r, {11, 22, 1}));
}

// --------------------------------------------------
// Entry
// --------------------------------------------------

int main() {
    test_vec2();
    test_vec3();
    test_vec4();

    test_mat3_identity();
    test_mat3_determinant();
    test_mat3_inverse();
    test_normal_matrix();

    test_mat4_identity();
    test_mat4_translation();
    test_mat4_translation_composition();

    test_affine_2d_transform();

    std::cout << "All geometry tests passed ✅\n";
    return 0;
}
