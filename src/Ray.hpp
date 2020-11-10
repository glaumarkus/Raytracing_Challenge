#ifndef RAY_HPP
#define RAY_HPP

#include "Vec4.hpp"
#include "Matrix.hpp"

class Ray {
private:
    Vec4 origin;
    Vec4 direction;
public:
    __host__ __device__ Ray() {
        origin = Vec4(0, 0, 0, 1);
        direction = Vec4(1, 0, 0, 0);
    }

    __host__ __device__ Ray(const Ray& r) :
        origin(r.origin),
        direction(r.direction)
    {};

    __host__ __device__ Ray(const Vec4& origin, const Vec4& direction) :
        origin(origin),
        direction(direction)
    {};

    __host__ __device__ void transform(Mat4& m) {
        origin = m * origin;
        direction = m * direction;
    }

    __host__ __device__ Vec4 getOrigin() {
        return origin;
    }

    __host__ __device__ Vec4 getDirection() {
        return direction;
    }

    __host__ __device__ Vec4 position(const float& t) const {
        return Vec4(origin + direction * t);
    };

    __host__ __device__ Ray& operator =(const Ray& r) {
        origin = r.origin;
        direction = r.direction;
        return *this;
    };

    __host__ __device__ void print() {
        printf("Ray: (%f, %f, %f, %f) (%f, %f, %f, %f)\n", origin.x, origin.y, origin.z, origin.w, direction.x, direction.y, direction.z, direction.w);
    }
};

#endif 