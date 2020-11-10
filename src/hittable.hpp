#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"


/*
HITTABLE - ABSTRACT
*/

class hittable {
public:
	__host__ __device__ virtual int getID() = 0;
	__device__ virtual void intersect(Intersection& i, const Ray& ray) {};
	__device__ virtual Vec4 normal_at(const Vec4& v) = 0;
	__device__ virtual Material getMaterial() = 0;
};

/*
Plane
*/

class Plane {
private:


public:
	int id;
	int MatID;

	Mat4 transformation;
	Material mat;

	/*
	* CONSTRUCTOR
	*/
	__host__ __device__ Plane(int id, Mat4 transformation, int MatID) :
		id(id),
		transformation(transformation),
		MatID(MatID)
	{}

	/*
	* NORMAL_AT
	*/
	__device__ Vec4 normal_at(const Vec4& v) {
		return Vec4(0,0,1,0);
	}

	/*
	* MATERIAL
	*/
	__device__ Material getMaterial() {
		return mat;
	}
	__device__ int getMatID() {
		return MatID;
	}

	/*
	* INTERSECT
	*/
	__device__ void intersect(Intersection& i, const Ray& ray) {
		Mat4 in_trans = transformation;

		//in_trans.print();

		if (!inverse(in_trans)) {
			//printf("cant inverse\t");
			return;
		}
			

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();

		//dir.print();
		//ori.print();

		if (std::abs(dir.z) < EPSILON) {
			return;
		}

		float t = -ori.z / dir.z;
		//printf("t: %f\n", t);
		i.push(id, t, this);
	}

	__host__ __device__ Plane operator =(const Plane& p) {
		id = p.id;
		transformation = p.transformation;
		mat = p.mat;
		MatID = p.MatID;
		return *this;
	}
};


/*
Sphere
*/

class Sphere {
private:
	


public:

	int id;
	int MatID;

	Mat4 transformation;
	Material mat;

	/*
	* CONSTRUCTOR
	*/
	/*
	__host__ __device__ Sphere(int id, Mat4 transformation, Material mat) :
		id(id),
		transformation(transformation),
		mat(mat)
	{}
	*/
	__host__ __device__ Sphere(int id, Mat4 transformation, int MatID) :
		id(id),
		transformation(transformation),
		MatID(MatID)
	{}

	__device__ int getMatID() {
		return MatID;
	}

	/*
	* NORMAL_AT
	*/
	__device__ Vec4 normal_at(const Vec4& v) {
		
		Mat4 in_trans = transformation;
		
		if (!inverse(in_trans))
			return Vec4(1,0,0,0);
		
		Vec4 objPt = in_trans * v;
		Vec4 objNrm = objPt - Vec4(0, 0, 0, 1);
		in_trans.transpose();
		Vec4 worldNrm = in_trans * objNrm;
		worldNrm.w = 0.0f;
		worldNrm.norm();
		return worldNrm;
	}

	/*
	* MATERIAL
	*/
	__device__ Material getMaterial() {
		return mat;
	}
	

	__host__ __device__ void print() {
		printf("Sphere with id %d, matid %d\n", id, MatID);
		transformation.print();
	}
	
	__host__ __device__ Sphere operator =(const Sphere& s) {
		id = s.id;
		transformation = s.transformation;
		mat = s.mat;
		MatID = s.MatID;
		return *this;
	}
	

	/*
	* INTERSECT
	*/
	__device__ void intersect(Intersection& i, const Ray& ray) {

		Mat4 in_trans = transformation;

		if (!inverse(in_trans))
			return;

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

		//printf("Hit sphere with id: %d\n", id);
	}
};



#endif