#ifndef PLANE_HPP
#define PLANE_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"


/*
Plane
*/

class Plane {
private:
	int id;
	int MatID;
	Mat4 transformation;
	Mat4 in_trans;

public:
	/*
	* CONSTRUCTOR
	*/
	__host__ __device__ Plane(const int& id, const Mat4& transformation, const int& MatID) :
		id(id),
		transformation(transformation),
		MatID(MatID)
	{
		in_trans = transformation;
		if (!inverse(in_trans));
	}

	/*
	* NORMAL_AT
	*/
	__host__ __device__ Vec4 normal_at(const Vec4& v) {


		Vec4 objPt = in_trans * v;
		Vec4 objNrm(0, 0, 1, 0);
		in_trans.transpose();
		Vec4 worldNrm = in_trans * objNrm;
		worldNrm.w = 0.0f;
		worldNrm.norm();
		return worldNrm;
	}

	/*
	* MATERIAL
	*/
	__host__ __device__ int getMatID() {
		return MatID;
	}

	/*
	* INTERSECT
	*/
	__host__ __device__ void intersect(Intersection& i, const Ray& ray) {

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();


		if (std::abs(dir.z) < EPSILON) {
			return;
		}

		float t = -ori.z / dir.z;
		i.push(id, t, this);
	}

	__host__ __device__ Plane operator =(const Plane& p) {
		id = p.id;
		transformation = p.transformation;
		MatID = p.MatID;
		in_trans = p.in_trans;
		return *this;
	}
};

#endif