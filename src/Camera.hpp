#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vec4.hpp"
#include "Ray.hpp"
#include <cmath>

class Camera {
private:

	Vec4 origin;
	Vec4 forward, right, up;
	float h, w;

public:

	__host__ Camera() {};

	__host__ void createCamera(const Vec4& originIN,
		const Vec4& forwardIN,
		const Vec4& upguideIN,
		float fovIN,
		float aspectRatioIN) {

		origin = originIN;
		forward = forwardIN;

		right = cross(forward, upguideIN);
		right.norm();

		up = cross(right, forward);
		up.norm();

		h = tan(fovIN);
		w = h * aspectRatioIN;
	}

	__host__ __device__ Camera(
		const Vec4& origin,
		const Vec4& forward,
		const Vec4& upguide,
		float fov,
		float aspectRatio) :
		origin(origin),
		forward(forward)
	{
		right = cross(forward, upguide);
		right.norm();

		up = cross(right, forward);
		up.norm();

		h = tan(fov);
		w = h * aspectRatio;
	};

	__host__ __device__ Ray getRay(const float& x, const float& y) {
		Vec4 direction = forward + x * w * right + y * h * up;
		direction.norm();
		return Ray(origin, direction);
	};
};

#endif // !CAMERA_H