#ifndef COLOR_HPP
#define COLOR_HPP

#include <cmath>
#include <algorithm>

class Color {
public:
	float r, g, b, a;

	__host__ __device__ Color() :
		r(0.0f),
		g(0.0f),
		b(0.0f)
	{
		a = 0.0f;
	};

	__host__ __device__ Color(float l) :
		r(l),
		g(l),
		b(l)
	{
		a = 0.0f;
	};

	__host__ __device__ Color(float r, float g, float b) :
		r(r),
		g(g),
		b(b)
	{
		a = 0.0f;
	};

	__host__ __device__ Color(float r, float g, float b, float a) :
		r(r),
		g(g),
		b(b),
		a(a)
	{};

	__host__ __device__ float getTransparency() {
		return a;
	}

	__host__ __device__ void clamp() {
		r = r > 0.0f ? (r > 1 ? 1.0f : r) : 0.0f;
		g = g > 0.0f ? (g > 1 ? 1.0f : g) : 0.0f;
		b = b > 0.0f ? (b > 1 ? 1.0f : b) : 0.0f;
	};

	__host__ __device__ Color& operator =(const Color& c) {
		r = c.r;
		g = c.g;
		b = c.b;
		a = c.a;
		return *this;
	};

	__host__ __device__ Color& operator +=(const Color& c) {
		r += c.r;
		g += c.g;
		b += c.b;
		return *this;
	};

	__host__ __device__ Color& operator *=(const Color& c) {
		r *= c.r;
		g *= c.g;
		b *= c.b;
		return *this;
	};

	__host__ __device__ Color& operator *=(float f) {
		r *= f;
		g *= f;
		b *= f;
		return *this;
	};

};

__host__ __device__ inline Color operator +(const Color& c1, const Color& c2)
{
	return Color(c1.r + c2.r,
		c1.g + c2.g,
		c1.b + c2.b);
}

__host__ __device__ inline Color operator *(const Color& c1, const Color& c2)
{
	return Color(c1.r * c2.r,
		c1.g * c2.g,
		c1.b * c2.b);
}

__host__ __device__ inline Color operator *(const Color& c, float f)
{
	return Color(c.r * f,
		c.g * f,
		c.b * f);
}

__host__ __device__ inline Color operator *(float f, const Color& c)
{
	return Color(f * c.r,
		f * c.g,
		f * c.b);
}


#endif // !COLOR_HPP