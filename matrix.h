#pragma once
#include <cmath>
#include <cassert>
#include <algorithm>
#include "vector.h"

template<int rows, int cols>
struct mat {
    static_assert(rows > 0 && cols > 0);
    vec<cols> rows_data[rows]{};

    constexpr vec<cols>& operator[] (const int idx) {
        assert(idx >= 0 && idx < rows);
        return rows_data[idx];
    }

    constexpr const vec<cols>& operator[] (const int idx) const {
        assert(idx >= 0 && idx < rows);
        return rows_data[idx];
    }

    constexpr static mat identity() {
        static_assert(rows == cols, "identity() requires a square matrix");

        mat r{};
        for (int i = 0; i < rows; ++i)
            r.rows_data[i][i] = 1.0;
        return r;
    }

    mat<rows - 1, cols - 1> minor(int row, int col) const {
        static_assert(rows > 1 && cols > 1, "minor() requires rows, cols > 1");
        assert(row >= 0 && row < rows);
        assert(col >= 0 && col < cols);

        mat<rows - 1, cols - 1> r{};

        int ri{};

        for (int i = 0; i < rows; ++i) {
            if (i == row) continue;

            int rj{};
            for (int j = 0; j < cols; j++) {
                if (j == col) continue;

                r[ri][rj++] = rows_data[i][j]; //surviving elements
            }
            ++ri;
        }
        return r;
    }

    double det() const {
        static_assert(rows == cols, "determinant requires square matrix");

        if constexpr (rows == 1) {
            return rows_data[0][0];
        }
        else if constexpr (rows == 2) {
            return
                rows_data[0][0] * rows_data[1][1] -
                rows_data[0][1] * rows_data[1][0];
        }
        else {
            double d = 0.0;
            for (int c = 0; c < cols; ++c)
                d += rows_data[0][c] * cofactor(0, c);
            return d;
        }
    }
    double cofactor(int row, int col) const {
        double d = this->minor(row, col).det();
        return ((row + col) & 1) ? -d : d; //“Is the least significant bit of (row + col) equal to 1?”
    }

    mat<rows, cols> invert_transpose() const { //normal matrix
        static_assert(rows == cols, "invert_transpose() requires a square matrix");
        double d = det();
        assert(std::abs(d) > 1e-8);
        
        mat<rows, cols> r{};
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                r[j][i] = cofactor(i, j) / d;
        return r;
    }

    constexpr mat<cols, rows> transpose() const { //coord space change
        mat<cols, rows> t{};
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                t[j][i] = rows_data[i][j];
        return t;
    }

};

//rows, cols
using mat2 = mat<2, 2>;
using mat3 = mat<3, 3>;
using mat4 = mat<4, 4>;

using mat2_3 = mat<2, 3>;
using mat2_4 = mat<2, 4>;

using mat3_2 = mat<3, 2>;
using mat3_4 = mat<3, 4>;

using mat4_2 = mat<4, 2>;
using mat4_3 = mat<4, 3>;


// -------- Operator Overloads ----------

//MATRIX WITH MATRIX
template<int R, int C, int K>
constexpr mat<R, C> operator*(const mat<R, K>& A, const mat<K, C>& B) {
    mat<R, C> Rm{};
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            for (int k = 0; k < K; ++k)
                Rm[i][j] += A[i][k] * B[k][j];
    return Rm;
}

template<int R, int C, int K>
mat<R, C> operator/(const mat<R, K>& A, const mat<K, C>& B) {
    mat<R, C> Rm{};
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            for (int k = 0; k < K; ++k) {
                assert(B[k][j] != 0);
                Rm[i][j] += A[i][k] / B[k][j];
            }
    return Rm;
}


//MATRIX WITH VECTOR

template<int R, int C>
constexpr vec<R> operator*(const mat<R, C>& M, const vec<C>& v) {
    vec<R> result{};
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            result[i] += M[i][j] * v[j];
    return result;
}

template<int R, int C>
constexpr vec<C> operator*(const vec<R>& v, const mat<R, C>& M) {
    vec<C> result{};
    for (int j = 0; j < C; ++j)
        for (int i = 0; i < R; ++i)
            result[j] += v[i] * M[i][j];
    return result;
}

template<int R, int C>
vec<R> operator/(const mat<R, C>& M, const vec<C>& v) {
    vec<R> result{};
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            assert(v[j] != 0);
            result[i] += M[i][j] / v[j];
        }
    }
    return result;
}

template<int R, int C>
vec<R> operator/(const vec<C>& v, const mat<R, C>& M) {
    vec<R> result{};
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) {
            assert(v[j] != 0);
            result[i] += M[i][j] / v[j];
        }
    return result;
}
