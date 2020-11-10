#ifndef CUBE_HPP
#define CUBE_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"


/*
Cube
*/

class Cube {
private:

	int id;
	int MatID;

	Mat4 transformation;
	Mat4 in_trans;

	struct t_helper {
		float t1, t2;
	};

	__host__ __device__ t_helper check_axis(const float& origin, const float& direction) {

		t_helper t;

		float tmin_num = (-1 - origin);
		float tmax_num = (1 - origin);

		if (std::abs(direction) >= EPSILON) {
			t.t1 = tmin_num / direction;
			t.t2 = tmax_num / direction;
		}
		else {
			t.t1 = tmin_num * INFINITY;
			t.t2 = tmax_num * INFINITY;
		}

		if (t.t1 > t.t2) {
			float tmp = t.t1;
			t.t1 = t.t2;
			t.t2 = tmp;
		}

		return t;
	}
	// max function
	__host__ __device__ float max3(const float& f1, const float& f2, const float& f3) {
		if (f1 > f2) {
			if (f1 > f3) {
				return f1;
			}
			return f3;
		}

		if (f2 > f3) {
			return f2;
		}
		return f3;
	}
	// min function
	__host__ __device__ float min3(const float& f1, const float& f2, const float& f3) {
		if (f1 < f2) {
			if (f1 < f3) {
				return f1;
			}
			return f3;
		}

		if (f2 < f3) {
			return f2;
		}
		return f3;
	}

public:



	/*
	* CONSTRUCTOR
	*/
	__host__ __device__ Cube() {}

	__host__ __device__ Cube(const int& id, const Mat4& transformation, const int& MatID) :
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

		Vec4 normal(0, 0, 0, 0);
		float ax, ay, az;
		ax = std::abs(objPt.x);
		ay = std::abs(objPt.y);
		az = std::abs(objPt.z);
		float maxv = max3(ax, ay, az);

		if (maxv == ax) {
			normal.x = objPt.x;
		}
		else if (maxv == ay) {
			normal.y = objPt.y;
		}
		else {
			normal.z = objPt.z;
		}

		in_trans.transpose();
		Vec4 worldNrm = in_trans * normal;
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

		Mat4 in_trans = transformation;
		if (!inverse(in_trans)) {
			return;
		}

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();

		t_helper xt = check_axis(ori.x, dir.x);
		t_helper yt = check_axis(ori.y, dir.y);
		t_helper zt = check_axis(ori.z, dir.z);

		float t1, t2;

		t1 = max3(xt.t1, yt.t1, zt.t1);
		t2 = min3(xt.t2, yt.t2, zt.t2);

		if (t1 > t2)
			return;

		i.push(id, t1, this);
		i.push(id, t2, this);

		// new push
		//i.push(id, t1, 2, INFINITY, INFINITY);
		//i.push(id, t2, 2, INFINITY, INFINITY);
	}

	__host__ __device__ bool intersect(const Ray& ray) {

		Mat4 in_trans = transformation;
		if (!inverse(in_trans)) {
			return false;
		}

		Ray r = ray;
		r.transform(in_trans);

		Vec4 dir = r.getDirection();
		Vec4 ori = r.getOrigin();

		t_helper xt = check_axis(ori.x, dir.x);
		t_helper yt = check_axis(ori.y, dir.y);
		t_helper zt = check_axis(ori.z, dir.z);

		float t1, t2;

		t1 = max3(xt.t1, yt.t1, zt.t1);
		t2 = min3(xt.t2, yt.t2, zt.t2);

		if (t1 > t2)
			return false;

		return true;
	}

	__host__ __device__ Cube operator =(const Cube& c) {
		id = c.id;
		transformation = c.transformation;
		MatID = c.MatID;
		in_trans = c.in_trans;
		return *this;
	}
};


#endif