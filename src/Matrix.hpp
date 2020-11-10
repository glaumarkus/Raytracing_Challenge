#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Vec4.hpp"


// Mat2
class Mat2 {
public:

    float f[2][2];

    __host__ __device__ Mat2() {
        f[0][0] = 1.0f;
        f[0][1] = 0.0f;

        f[1][1] = 1.0f;
        f[1][0] = 0.0f;
    }

    __host__ __device__ Mat2(const float fi[2][2]) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                f[r][c] = fi[r][c];
            }
        }
    }

    __host__ __device__ Mat2(const Mat2& m2) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                f[r][c] = m2.f[r][c];
            }
        }
    }

    __host__ __device__ void transpose() {

        float tmp;

        // swapping 1 cells
        tmp = f[0][1];
        f[0][1] = f[1][0];
        f[1][0] = tmp;
    }

    __host__ __device__ inline Mat2 operator =(const Mat2& m2) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                f[r][c] = m2.f[r][c];
            }
        }
        return *this;
    }

};

// determinante
__host__ __device__ inline float determinant(const Mat2& m2) {
    return m2.f[0][0] * m2.f[1][1] - m2.f[1][0] * m2.f[0][1];
}



// Mat3
class Mat3 {
public:

    float f[3][3];

    __host__ __device__ Mat3() {
        f[0][0] = 1.0f;
        f[0][1] = 0.0f;
        f[0][1] = 0.0f;

        f[1][0] = 0.0f;
        f[1][1] = 1.0f;
        f[1][2] = 0.0f;

        f[2][0] = 0.0f;
        f[2][1] = 0.0f;
        f[2][2] = 1.0f;
    }

    __host__ __device__ Mat3(const float fi[3][3]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                f[r][c] = fi[r][c];
            }
        }
    }

    __host__ __device__ Mat3(const Mat3& m3) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                f[r][c] = m3.f[r][c];
            }
        }
    }

    __host__ __device__ inline Mat3 operator =(const Mat3& m3) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                f[r][c] = m3.f[r][c];
            }
        }
        return *this;
    }
};

// Mat3 submatrix
__host__ __device__ Mat2 submatrix(const Mat3& m3, const int& ri, const int& ci) {

    float f[2][2];

    int nr = 0;


    for (int r = 0; r < 3; ++r) {
        if (r == ri)
            continue;

        int nc = 0;
        for (int c = 0; c < 3; ++c) {
            if (c == ci)
                continue;

            f[nr][nc] = m3.f[r][c];
            nc++;
        }
        nr++;
    }

    return Mat2(f);
}

__host__ __device__ float minor(const Mat3& m3, const int& ri, const int& ci) {
    Mat2 m2(submatrix(m3, ri, ci));
    return determinant(m2);
}

__host__ __device__ float cofactor(const Mat3& m3, const int& ri, const int& ci) {
    return minor(m3, ri, ci) * ((ri + ci) % 2 > 0 ? -1 : 1);
}

__host__ __device__ float determinant(const Mat3& m3) {
    float det = 0.0f;
    for (int c = 0; c < 3; ++c) {
        det += m3.f[0][c] * cofactor(m3, 0, c);
    }
    return det;
}


// Mat4
class Mat4 {
public:

    float f[4][4];

    __host__ __device__ Mat4() {
        f[0][0] = 1.0f;
        f[0][1] = 0.0f;
        f[0][2] = 0.0f;
        f[0][3] = 0.0f;

        f[1][0] = 0.0f;
        f[1][1] = 1.0f;
        f[1][2] = 0.0f;
        f[1][3] = 0.0f;

        f[2][0] = 0.0f;
        f[2][1] = 0.0f;
        f[2][2] = 1.0f;
        f[2][3] = 0.0f;

        f[3][0] = 0.0f;
        f[3][1] = 0.0f;
        f[3][2] = 0.0f;
        f[3][3] = 1.0f;
    }

    __host__ __device__ Mat4(const float fi[4][4]) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                f[r][c] = fi[r][c];
            }
        }
    }

    __host__ __device__ Mat4(const Mat4& m4) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                f[r][c] = m4.f[r][c];
            }
        }
    }

    __host__ __device__ void transpose() {

        float tmp;

        // swapping 6 cells
        tmp = f[0][1];
        f[0][1] = f[1][0];
        f[1][0] = tmp;

        tmp = f[0][2];
        f[0][2] = f[2][0];
        f[2][0] = tmp;

        tmp = f[0][3];
        f[0][3] = f[3][0];
        f[3][0] = tmp;

        tmp = f[1][2];
        f[1][2] = f[2][1];
        f[2][1] = tmp;

        tmp = f[1][3];
        f[1][3] = f[3][1];
        f[3][1] = tmp;

        tmp = f[2][3];
        f[2][3] = f[3][2];
        f[3][2] = tmp;

    }

    __host__ __device__ inline Mat4 operator =(const Mat4& m4) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                f[r][c] = m4.f[r][c];
            }
        }
        return *this;
    }

    __host__ __device__ void print() {
        printf("(%f, %f, %f, %f)\n", f[0][0], f[0][1], f[0][2], f[0][3]);
        printf("(%f, %f, %f, %f)\n", f[1][0], f[1][1], f[1][2], f[1][3]);
        printf("(%f, %f, %f, %f)\n", f[2][0], f[2][1], f[2][2], f[2][3]);
        printf("(%f, %f, %f, %f)\n", f[3][0], f[3][1], f[3][2], f[3][3]);
    }
};

// Mat4 submatrix
__host__ __device__ Mat3 submatrix(const Mat4& m4, const int& ri, const int& ci) {

    float f[3][3];

    int nr = 0;

    for (int r = 0; r < 4; ++r) {
        if (r == ri)
            continue;
        int nc = 0;
        for (int c = 0; c < 4; ++c) {
            if (c == ci)
                continue;

            f[nr][nc] = m4.f[r][c];
            nc++;
        }
        nr++;
    }

    return Mat3(f);
}

__host__ __device__ float determinant(const Mat4& m4) {
    float det = 0.0f;
    for (int c = 0; c < 4; ++c) {
        Mat3 sub(submatrix(m4, 0, c));
        float tmp = 0.0f;
        for (int c2 = 0; c2 < 3; ++c2) {
            tmp += sub.f[0][c2] * cofactor(sub, 0, c2);
        }
        det += tmp * (c % 2 > 0 ? -1 : 1) * m4.f[0][c];
    }
    return det;
}

__host__ __device__ bool inverse(Mat4& m4) {
    float det = determinant(m4);
    if (det < EPSILON && det > -EPSILON)
        return false;

    float m4_new[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float tmp = determinant(submatrix(m4, r, c)) * ((r + c) % 2 > 0 ? -1 : 1);
            m4_new[c][r] = tmp / det;
            //m4_new[r][c] = tmp;
        }
    }
    m4 = Mat4(m4_new);
    return true;
}


// equal
__host__ __device__ bool equal(const Mat2& m1, const Mat2& m2) {
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            if (m1.f[r][c] != m2.f[r][c])
                return false;
        }
    }
    return true;
}

__host__ __device__ bool equal(const Mat3& m1, const Mat3& m2) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (m1.f[r][c] != m2.f[r][c])
                return false;
        }
    }
    return true;
}

__host__ __device__ bool equal(const Mat4& m1, const Mat4& m2) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (m1.f[r][c] != m2.f[r][c])
                return false;
        }
    }
    return true;
}


// addition
__host__ __device__ inline Mat2 operator +(const Mat2& m1, const Mat2& m2) {
    float f[2][2];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            f[r][c] = m1.f[r][c] + m2.f[r][c];
        }
    }
    return Mat2(f);
}

__host__ __device__ inline Mat3 operator +(const Mat3& m1, const Mat3& m2) {
    float f[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            f[r][c] = m1.f[r][c] + m2.f[r][c];
        }
    }
    return Mat3(f);
}

__host__ __device__ inline Mat4 operator +(const Mat4& m1, const Mat4& m2) {
    float f[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            f[r][c] = m1.f[r][c] + m2.f[r][c];
        }
    }
    return Mat4(f);
}

// substraction
__host__ __device__ inline Mat2 operator -(const Mat2& m1, const Mat2& m2) {
    float f[2][2];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            f[r][c] = m1.f[r][c] - m2.f[r][c];
        }
    }
    return Mat2(f);
}

__host__ __device__ inline Mat3 operator -(const Mat3& m1, const Mat3& m2) {
    float f[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            f[r][c] = m1.f[r][c] - m2.f[r][c];
        }
    }
    return Mat3(f);
}

__host__ __device__ inline Mat4 operator -(const Mat4& m1, const Mat4& m2) {
    float f[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            f[r][c] = m1.f[r][c] - m2.f[r][c];
        }
    }
    return Mat4(f);
}

// multiplication
__host__ __device__ inline Mat2 operator *(const Mat2& m1, const Mat2& m2) {
    float f[2][2];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            float tmp = 0.0f;
            for (int i = 0; i < 2; ++i) {
                tmp += m1.f[r][i] * m2.f[i][c];
            }
            f[r][c] = tmp;
        }
    }
    return Mat2(f);
}

__host__ __device__ inline Mat3 operator *(const Mat3& m1, const Mat3& m2) {
    float f[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float tmp = 0.0f;
            for (int i = 0; i < 3; ++i) {
                tmp += m1.f[r][i] * m2.f[i][c];
            }
            f[r][c] = tmp;
        }
    }
    return Mat3(f);
}

__host__ __device__ inline Mat4 operator *(const Mat4& m1, const Mat4& m2) {
    float f[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float tmp = 0.0f;
            for (int i = 0; i < 4; ++i) {
                tmp += m1.f[r][i] * m2.f[i][c];
            }
            f[r][c] = tmp;
        }
    }
    return Mat4(f);
}

// multiplication vector
__host__ __device__ inline Vec4 operator *(const Mat4& m, const Vec4& v) {
    Vec4 copy(v);
    float f[4];
    for (int r = 0; r < 4; ++r) {
        float tmp = 0.0f;
        for (int c = 0; c < 4; ++c) {
            tmp += m.f[r][c] * copy.idx(c);
        }
        f[r] = tmp;
    }
    return Vec4(f[0], f[1], f[2], f[3]);
}


// move pt
__host__ __device__ Mat4 translate(const float& x, const float& y, const float& z) {
    float f[4][4] = {
        {1.0f, 0.0f, 0.0f, x   },
        {0.0f, 1.0f, 0.0f, y   },
        {0.0f, 0.0f, 1.0f, z   },
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    return Mat4(f);
}

// increase pt
__host__ __device__ Mat4 scale(const float& x, const float& y, const float& z) {
    float f[4][4] = {
        {x,    0.0f, 0.0f, 0.0f},
        {0.0f, y,    0.0f, 0.0f},
        {0.0f, 0.0f, z,    0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    return Mat4(f);
}

// rotate pt on x
__host__ __device__ Mat4 rotate_x(const float& rad) {
    float f[4][4] = {
        {1.0f, 0.0f,      0.0f,     0.0f},
        {0.0f, cos(rad), -sin(rad), 0.0f},
        {0.0f, sin(rad),  cos(rad), 0.0f},
        {0.0f, 0.0f,      0.0f,     1.0f}
    };
    return Mat4(f);
}

// rotate pt on y
__host__ __device__ Mat4 rotate_y(const float& rad) {
    float f[4][4] = {
        {cos(rad),  0.0f, sin(rad), 0.0f},
        {0.0f,      1.0f, 0.0f,     0.0f},
        {-sin(rad), 0.0f, cos(rad), 0.0f},
        {0.0f,      0.0f, 0.0f,     1.0f}
    };
    return Mat4(f);
}

// rotate pt on z
__host__ __device__ Mat4 rotate_z(const float& rad) {
    float f[4][4] = {
        {cos(rad), -sin(rad),  0.0f, 0.0f},
        {sin(rad),  cos(rad),  0.0f, 0.0f},
        {0.0f,      0.0f,      1.0f, 0.0f},
        {0.0f,      0.0f,      0.0f, 1.0f}
    };
    return Mat4(f);
}

// influence of params on others
__host__ __device__ Mat4 shear(const float& xy, const float& xz, const float& yx, const float& yz, const float& zx, const float& zy) {
    float f[4][4] = {
        {1.0f, xy,   xz,   0.0f},
        {yx,   1.0f, yz,   0.0f},
        {zx,   zy,   1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    return Mat4(f);
}

// chain all transformations
__host__ __device__ Mat4 transform(
    const float& x_translate = 1.0f,
    const float& y_translate = 1.0f,
    const float& z_translate = 1.0f,
    const float& x_scale = 1.0f,
    const float& y_scale = 1.0f,
    const float& z_scale = 1.0f,
    const float& x_rotate = 1.0f,
    const float& y_rotate = 1.0f,
    const float& z_rotate = 1.0f,
    const float& xy = 1.0f,
    const float& xz = 1.0f,
    const float& yx = 1.0f,
    const float& yz = 1.0f,
    const float& zx = 1.0f,
    const float& zy = 1.0f
) {
    Mat4 translation, scaling, rotation_x, rotation_y, rotation_z, shearing;

    if (x_translate != 1.0f || y_translate != 1.0f || y_translate == 1.0f)
        translation = translate(x_translate, y_translate, z_translate);

    if (x_scale != 1.0f || y_scale != 1.0f || z_scale != 1.0f)
        scaling = scale(x_scale, y_scale, z_scale);

    if (x_rotate != 1.0f)
        rotation_x = rotate_x(x_rotate);

    if (y_rotate != 1.0f)
        rotation_y = rotate_y(y_rotate);

    if (z_rotate != 1.0f)
        rotation_z = rotate_z(z_rotate);

    if (xy != 1.0f || xz != 1.0f || yx != 1.0f || yz != 1.0f || zx != 1.0f || zy != 1.0f)
        shearing = shear(xy, xz, yx, yz, zx, zy);

    return Mat4(translation * scaling * rotation_x * rotation_y * rotation_z * shearing);
}

#endif // !MATRIX_HPP