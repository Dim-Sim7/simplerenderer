#pragma once
#include <cuda_runtime.h>

//GPU MATH FUNCTIONS

__device__ inline float dot(float2 a, float2 b) {
    return a.x*b.x + a.y*b.y;
}
__device__ inline float dot(const float3 a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float invLength(const float3 v) {
    return rsqrtf(dot(v, v));
}

__device__ inline float3 normalize(const float3& v) {
    float invLen = invLength(v);
    return make_float3(
        v.x * invLen,
        v.y * invLen,
        v.z * invLen
    );
}
__device__ inline float3 cross(const float3& a, const float3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

//Operator overloads
__device__ inline float3 operator*(float3 a, float b) {
    return float3{a.x * b, a.y * b, a.z * b};
}

__device__ inline float3 operator*(float b, float3 a) {
    return float3{a.x * b, a.y * b, a.z * b};
}

__device__ inline float3 operator+(float3 a, float3 b) {
    return float3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline float3 operator-(float3 a, float3 b) {
    return float3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline float3 operator-(float3 a, float b) {
    return float3{a.x - b, a.y - b, a.z - b};
}


__host__ __device__ inline float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

