#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"

/*
Sphere
*/

class Sphere  {
private:
	int id;
	int MatID;
	Mat4 transformation;
	Mat4 in_trans;

public:


	/*
	* CONSTRUCTOR
	*/
	__host__ __device__ Sphere(const int& id, const Mat4& transformation, const int& MatID) :
		id(id),
		transformation(transformation),
		MatID(MatID)
	{
		in_trans = transformation;
		if (!inverse(in_trans));
	}

	/*
	* MATERIAL
	*/
	__host__ __device__ int getMatID() {
		return MatID;
	}

	/*
	* NORMAL_AT
	*/
	__host__ __device__ Vec4 normal_at(const Vec4& v) {

		Vec4 objPt = in_trans * v;
		Vec4 objNrm = objPt - Vec4(0, 0, 0, 1);
		in_trans.transpose();
		Vec4 worldNrm = in_trans * objNrm;
		worldNrm.w = 0.0f;
		worldNrm.norm();
		return worldNrm;
	}

	/*
	* INTERSECT
	*/
	__host__ __device__ void intersect(Intersection& i, const Ray& ray) {

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();

		Vec4 SphereToRay(ori - Vec4(0, 0, 0, 1));

		float a, b, c;

		a = dot(dir, dir);
		b = 2 * dot(dir, SphereToRay);
		c = dot(SphereToRay, SphereToRay) - 1;

		float discriminant = b * b - 4.0f * a * c;

		if (discriminant < 0)
			return;

		float t1, t2;
		t1 = (-b - std::sqrt(discriminant)) / (2.0f * a);
		t2 = (-b + std::sqrt(discriminant)) / (2.0f * a);


		if (t1 == t2) {
			i.push(id, t1, this);
			return;
		}

		i.push(id, t1, this);
		i.push(id, t2, this);

	}

	// generic
	__host__ __device__ void print() {
		printf("Sphere with id %d, matid %d\n", id, MatID);
		transformation.print();
	}

	__host__ __device__ Sphere operator =(const Sphere& s) {
		id = s.id;
		transformation = s.transformation;
		MatID = s.MatID;
		in_trans = s.in_trans;
		return *this;
	}

};

#endif