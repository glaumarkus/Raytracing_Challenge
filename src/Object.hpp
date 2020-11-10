#ifndef OBJECT_HPP
#define OBJECT_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"


class Object {
private:
	int id_;
	Mat4 transformation_;
	Mat4 inverse_transformation_;
	int matId_;
public:
	__host__ __device__ virtual int getMatID() = 0;
	__host__ __device__ virtual Vec4 normal_at(const Vec4& v, const float& uIN, const float& vIN) = 0;
	__host__ __device__ virtual void intersect(Intersection& i, const Ray& ray) = 0;
	//__host__ __device__ bool intersect(Intersection& i, const Ray& ray) = 0;
};


#endif
