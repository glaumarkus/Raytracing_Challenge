#ifndef VEC_HPP
#define VEC_HPP

// includes
#include <cmath>
#include <iostream>


#ifndef PI
#define PI 3.1415926f
#endif

#ifndef EPSILON
#define EPSILON 0.0001
#endif

#ifndef INFINITY
#define INFINITY 1e9
#endif



class Vec4 {
public:

	// vars
	float x, y, z, w;

	__host__ __device__ void print() {
		printf("Vec: (%f, %f, %f, %f)\n", x, y, z, w);
	}

	// constructors
	__host__ __device__ Vec4() {
		x = 1.0f;
		y = 0.0f;
		z = 0.0f;
		w = 0.0f;
	}

	__host__ __device__ Vec4(const float& x, const float& y, const float& z) :
		x(x),
		y(y),
		z(z)
	{
		w = 0.0f;
	}

	__host__ __device__ Vec4(const float& x, const float& y, const float& z, const float& w) :
		x(x),
		y(y),
		z(z),
		w(w) {}

	__host__ __device__ Vec4(const Vec4& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;
	}

	__host__ __device__ inline float length2() {
		float f = x * x + y * y + z * z + w * w;
		return f;
	}

	__host__ __device__ inline float length() {
		float f = std::sqrt(length2());
		return f;
	}

	__host__ __device__ void norm() {
		float oneOverLength = 1.f / length();
		x *= oneOverLength;
		y *= oneOverLength;
		z *= oneOverLength;
		w *= oneOverLength;
	}


	// operators
	__host__ __device__ Vec4& operator =(const Vec4& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;
		return *this;
	}

	__host__ __device__ Vec4& operator +=(const Vec4& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	__host__ __device__ Vec4& operator -=(const Vec4& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}
	__host__ __device__ Vec4& operator *=(float f) {
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		return *this;
	}
	__host__ __device__ Vec4& operator /=(float f) {
		x /= f;
		y /= f;
		z /= f;
		w /= f;
		return *this;
	}

	__host__ __device__ float idx(int idx) {
		if (idx == 0)
			return x;
		if (idx == 1)
			return y;
		if (idx == 2)
			return z;
		if (idx == 3)
			return w;
		else
			return 0.0f;
	}
};


// add operators
__host__ __device__ inline Vec4 operator +(const Vec4& v1, const Vec4& v2)
{
	return Vec4(v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z,
		v1.w + v2.w);
}

__host__ __device__ inline Vec4 operator -(const Vec4& v1, const Vec4& v2)
{
	return Vec4(v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z,
		v1.w - v2.w);
}

__host__ __device__ inline Vec4 operator *(const Vec4& v1, const Vec4& v2)
{
	return Vec4(v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z,
		v1.w * v2.w);
}

__host__ __device__ inline Vec4 operator *(const Vec4& v, float f)
{
	return Vec4(v.x * f,
		v.y * f,
		v.z * f,
		v.w * f);
}

__host__ __device__ inline Vec4 operator *(float f, const Vec4& v)
{
	return Vec4(f * v.x,
		f * v.y,
		f * v.z,
		f * v.w);
}

__host__ __device__ inline Vec4 operator /(const Vec4& v1, const Vec4& v2)
{
	return Vec4(v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z,
		v1.w / v2.w);
}

__host__ __device__ inline Vec4 operator /(const Vec4& v, float f)
{
	return Vec4(v.x / f,
		v.y / f,
		v.z / f,
		v.w / f);
}

__host__ __device__ inline Vec4 operator /(float f, const Vec4& v)
{
	return Vec4(f / v.x,
		f / v.y,
		f / v.z,
		f / v.w);
}

// Vector operations

// dot
__host__ __device__ float dot(const Vec4& v1, const Vec4& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

// cross
__host__ __device__ Vec4 cross(const Vec4& v1, const Vec4& v2) {
	return
		Vec4(v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x);
}

// reflect
__host__ __device__ Vec4 reflect(const Vec4& v1, const Vec4& v2) {
	return Vec4(v1 - v2 * 2.0f * dot(v1, v2));
}

// basic operations

__host__ __device__ bool equal(const float& f1, const float& f2) {
	if (std::abs(f1 - f2) < EPSILON)
		return true;
	return false;
}

class Point : public Vec4 {
};


#endif