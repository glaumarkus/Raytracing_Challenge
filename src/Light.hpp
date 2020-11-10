#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <cmath>

#include "Material.hpp"
#include "Vec4.hpp"
#include "Color.hpp"


// TODO: Soft Shadows

class Light {


public:

	Color intensity;
	Vec4 position;
	int pts;
	float radius;

	__host__ __device__ Light() {}

	__host__ __device__ Light(const Color& intensity, const Vec4& position) :
		intensity(intensity),
		position(position)
	{
		pts = 20;
		radius = 2.5f;
	}

	__host__ __device__ Light* operator =(const Light* l) {
		intensity = l->intensity;
		position = l->position;
		pts = l->pts;
		return this;
	}

	__host__ __device__ int getNumPts() {
		return pts;
	}

	__host__ __device__ Vec4 getPt(int i) {


		float phi = PI * (3.0f - std::sqrt(5.0f));
		float x, y, z;

		float h = (float)i / (pts - 1);
		h *= radius;
		y = 1.0f - h;

		float tmp = 1 - y * y;
		float radius = std::sqrt(tmp);
		float t = phi * i;

		x = cos(t) * radius;
		z = sin(t) * radius;

		return Vec4(x, y, z) + position;
	}

};


#endif