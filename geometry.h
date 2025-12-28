#pragma once
#include <cmath>
#include <cassert>
#include <iostream>

struct Face {
    int v[3];
    int vt[3];
    int vn[3];
};


template<int n> struct vec {
    double data[n] = {0};
    double& operator[](const int i)       { assert(i>=0 && i<n); return data[i]; }
    double  operator[](const int i) const { assert(i>=0 && i<n); return data[i]; }
};

//OPERATOR OVERLOADS
template<int n> std::ostream& operator<<(std::ostream& out, const vec<n>& v) {
    for (int i=0; i<n; i++) out << v[i] << " ";
    return out;
}

template<int n>
double operator*(const vec<n>& lhs, const vec<n>& rhs) {
    double ret = 0.0;
    for (int i = 0; i < n; ++i) {
        ret += lhs[i] * rhs[i];
    }
    return ret;
}

template<int n>
vec<n> operator+(const vec<n>& lhs, const vec<n>& rhs) {
    vec<n> ret;
    for (int i = 0; i < n; ++i) {
        ret[i] = lhs[i] + rhs[i];
    }
    return ret;
}

template<int n>
vec<n> operator-(const vec<n>& lhs, const vec<n>& rhs) {
    vec<n> ret;
    for (int i = 0; i < n; ++i) {
        ret[i] = lhs[i] - rhs[i];
    }
    return ret;
}

template<int n>
vec<n> operator*(const vec<n>& lhs, const double& rhs) {
    vec<n> ret;
    for (int i = 0; i < n; i++) {
        ret[i] = lhs[i] * rhs;
    }
    return ret;
}

template<int n>
vec<n> operator*(const double& lhs, const vec<n>& rhs) {
    return rhs * lhs;
}

template<int n>
vec<n> operator/(const vec<n>& lhs, double rhs) {
    vec<n> ret;
    for (int i = 0; i < n; ++i) {
        ret[i] = lhs[i] / rhs;
    }
    return ret;
}


//TEMPLATE VECTORS
template<> struct vec<2> {
    double x = 0, y = 0;

    // non-const operator[]
    double& operator[](int i) {
        assert(i >= 0 && i < 2);
        return (i == 0) ? x : y;
    }

    // const operator[]
    const double& operator[](int i) const {
        assert(i >= 0 && i < 2);
        return (i == 0) ? x : y;
    }
};


template<>
struct vec<3> {
    double x = 0;
    double y = 0;
    double z = 0;

    double& operator[](int i) {
        assert(i >= 0 && i < 3);
        switch (i) {
            case 0: return x;
            case 1: return y;
            default: return z;
        }
    }

    const double operator[](int i) const {
        assert(i >= 0 && i < 3);
        switch (i) {
            case 0: return x;
            case 1: return y;
            default: return z;
        }
    }

    vec<2> xy() const { return {x, y}; }
    vec<2> xz() const { return {x, z}; }
    vec<2> yz() const { return {y, z}; }
};

template<>
struct vec<4> {
    double x = 0;
    double y = 0;
    double z = 0;
    double w = 0;

    double& operator[](int i) {
        assert(i >= 0 && i < 4);
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return w;
        }
    }

    const double operator[](int i) const {
        assert(i >= 0 && i < 4);
        switch (i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return w;
        }
    }

    vec<2> xy() const  { return {x, y}; }
    vec<3> xyz() const { return {x, y, z}; }
};


typedef vec<2> vec2;
typedef vec<3> vec3;
typedef vec<4> vec4;

template<int n> double norm(const vec<n>& v) {
    return std::sqrt(v);
}

template<int n> vec<n> normalized(const vec<n>& v) {
    return v / norm(v);
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x};
}

template<int n> struct dt;

template<int nrows, int ncols> struct mat {
    vec<ncols> rows[nrows] = {{}}; //initialise all values to 0

    vec<ncols>& operator[](int idx) {
        assert(idx >= 0 && idx < nrows);
        return rows[idx];
    }

    const vec<ncols>& operator[](int idx) const {
        assert(idx >= 0 && idx < nrows);
        return rows[idx];
    }


    double det() const {
        return dt<ncols>::det(*this);
    }

    double cofactor(const int row, const int col) const {
        mat<nrows-1,ncols-1> submatrix;
        for (int i=nrows-1; i--; )
            for (int j=ncols-1;j--; submatrix[i][j]=rows[i+int(i>=row)][j+int(j>=col)]);
        return submatrix.det() * ((row+col)%2 ? -1 : 1);
    }

    mat<nrows,ncols> invert_transpose() const {
        mat<nrows,ncols> adjugate_transpose; // transpose to ease determinant computation, check the last line
        for (int i=nrows; i--; )
            for (int j=ncols; j--; adjugate_transpose[i][j]=cofactor(i,j));
        return adjugate_transpose/(adjugate_transpose[0]*rows[0]);
    }

    mat<nrows,ncols> invert() const {
        return invert_transpose().transpose();
    }

    mat<ncols,nrows> transpose() const {
        mat<ncols,nrows> ret;
        for (int i=ncols; i--; )
            for (int j=nrows; j--; ret[i][j]=rows[j][i]);
        return ret;
    }

};

template<int nrows, int ncols>
vec<ncols> operator*(const vec<nrows>& lhs, const mat<nrows, ncols>& rhs)
{
    vec<ncols> result;

    for (int j = 0; j < ncols; ++j) {
        result[j] = 0;
        for (int i = 0; i < nrows; ++i)
            result[j] += lhs[i] * rhs[i][j];
    }

    return result;
}

template<int nrows, int ncols>
vec<nrows> operator*(const mat<nrows, ncols>& lhs, const vec<ncols>& rhs)
{
    vec<nrows> result;

    for (int i = 0; i < nrows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < ncols; ++j)
            sum += lhs[i][j] * rhs[j];
        result[i] = sum;
    }

    return result;
}

template<int R1, int C1, int C2>
mat<R1, C2> operator*(const mat<R1, C1>& lhs, const mat<C1, C2>& rhs)
{
    mat<R1, C2> result;

    for (int i = 0; i < R1; ++i) {
        for (int j = 0; j < C2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < C1; ++k)
                sum += lhs[i][k] * rhs[k][j];

            result[i][j] = sum;
        }
    }

    return result;
}

template<int nrows, int ncols>
mat<nrows, ncols> operator*(const mat<nrows, ncols>& lhs, double val)
{
    mat<nrows, ncols> result;

    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            result[i][j] = lhs[i][j] * val;

    return result;
}

template<int nrows, int ncols>
mat<nrows, ncols> operator/(const mat<nrows, ncols>& lhs, double val)
{
    mat<nrows, ncols> result;

    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            result[i][j] = lhs[i][j] / val;

    return result;
}

template<int nrows, int ncols>
mat<nrows, ncols> operator+(const mat<nrows, ncols>& lhs, const mat<nrows, ncols>& rhs)
{
    mat<nrows, ncols> result;

    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            result[i][j] = lhs[i][j] + rhs[i][j];

    return result;
}

template<int nrows, int ncols>
mat<nrows, ncols> operator-(const mat<nrows, ncols>& lhs, const mat<nrows, ncols>& rhs)
{
    mat<nrows, ncols> result;

    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
            result[i][j] = lhs[i][j] - rhs[i][j];

    return result;
}

template<int nrows, int ncols>
std::ostream& operator<<(std::ostream& out, const mat<nrows, ncols>& m)
{
    for (int i = 0; i < nrows; ++i)
        out << m[i] << '\n';

    return out;
}

template<int n>
struct dt {
    static double det(const mat<n, n>& src)
    {
        double sum = 0.0;

        for (int col = 0; col < n; ++col)
            sum += src[0][col] * src.cofactor(0, col);

        return sum;
    }
};

template<>
struct dt<1> {
    static double det(const mat<1, 1>& src)
    {
        return src[0][0];
    }
};







