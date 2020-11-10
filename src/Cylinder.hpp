#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "Plane.hpp"

/*
Sphere
*/

class Cylinder {
private:
	int id;
	int MatID;
	Mat4 transformation;
	Mat4 in_trans;

public:


	/*
	* CONSTRUCTOR
	*/
	__host__ __device__ Cylinder(const int& id, const Mat4& transformation, const int& MatID) :
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
		Vec4 objNrm = objPt;

		if (std::abs(objNrm.z) - 1 < 0.1) {
			objNrm = Vec4(0, 0, 1, 0);
		}
		else {
			objNrm.z = 0.0f;
		}

		in_trans.transpose();
		Vec4 worldNrm = in_trans * objNrm;
		worldNrm.w = 0.0f;
		worldNrm.norm();
		return worldNrm;
	}

	/*
	* INTERSECT
	*/
	__host__ __device__ float checkCap(const Ray& ray, Intersection& i) {

		Mat4 m1 = translate(0.0f, 0.0f, 1.0f);
		Mat4 m2 = translate(0.0f, 0.0f, -1.0f);

		Plane up(0, m1, 0);
		Plane down(0, m2, 0);

		up.intersect(i, ray);
		down.intersect(i, ray);

		return i.getT();
	}


	__host__ __device__ void intersect(Intersection& i, const Ray& ray) {

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();

		float a = dir.x * dir.x + dir.y * dir.y;

		if (a < EPSILON) {

			Intersection i;
			float t = checkCap(r, i);
			Vec4 pt(r.position(t));
			if ((pt.x * pt.x + pt.z * pt.z) <= 1.0f) {
				i.push(id, t, this);
			}
			return;
		}

		float b = 2 * ori.x * dir.x + 2 * ori.y * dir.y;
		float c = ori.x * ori.x + ori.y * ori.y - 1;

		float disc = b * b - 4 * a * c;

		if (disc < 0) {
			return;
		}

		float t1, t2;
		t1 = (-b - std::sqrt(disc)) / (2.0f * a);
		t2 = (-b + std::sqrt(disc)) / (2.0f * a);


		// check if within bounds of -1 - 1
		Vec4 pt1(r.position(t1));
		Vec4 pt2(r.position(t2));

		if (t1 == t2 && pt1.z <= 1.0f && pt1.z >= -1.0f) {
			i.push(id, t1, this);
			return;
		}

		if (pt1.z <= 1.0f && pt1.z >= -1.0f && pt2.z <= 1.0f && pt2.z >= -1.0f) {

			// if both t vals are between 1.0 no intersection with cap
			i.push(id, t1, this);
			i.push(id, t2, this);
		}
		else if (pt1.z <= 1.0f && pt1.z >= -1.0f) {
			// pt1 is correct, pt2 needs to be plane
			i.push(id, t1, this);
			Intersection i;
			float t = checkCap(r, i);
			i.push(id, t, this);
		}
		else if (pt2.z <= 1.0f && pt2.z >= -1.0f) {
			// pt2 is correct, pt1 needs to be plane
			i.push(id, t1, this);
			Intersection i;
			float t = checkCap(r, i);
			i.push(id, t, this);
		}
		/*

				if (pt1.z <= 1.0f && pt1.z >= -1.0f) {
					i.push(id, t1, this);
				}

				if (pt2.z <= 1.0f && pt2.z >= -1.0f) {
					i.push(id, t2, this);
				}

		*/


	}

	// generic
	__host__ __device__ void print() {
	}

	__host__ __device__ Cylinder operator =(const Cylinder& c) {
		id = c.id;
		transformation = c.transformation;
		MatID = c.MatID;
		in_trans = c.in_trans;
		return *this;
	}

};

#endif