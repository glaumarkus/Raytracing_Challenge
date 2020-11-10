#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <cmath>
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Material.hpp"

#include "Cube.hpp"
#include "CUDAVector.cuh"



class Triangle {
private:

	// normal tri
	int id;
	Vec4 p1, p2, p3;
	Vec4 e1, e2;
	Vec4 normal;
	int MatID;

	// smoothed tri
	bool isSmooth = false;
	Vec4 n1, n2, n3;

	// texture tri
	bool hastxt = false;
	Vec4 t1, t2, t3;

public:

	/*
	* Constructor
	*/

	__host__ __device__ Triangle() {}

	__host__ __device__ Triangle(const Triangle& other)	{
		id = other.id;
		MatID = other.MatID;
		isSmooth = other.isSmooth;
		hastxt = other.hastxt;
		p1 = other.p1;
		p2 = other.p2;
		p3 = other.p3;
		n1 = other.n1;
		n2 = other.n2;
		n3 = other.n3;
		t1 = other.t1;
		t2 = other.t2;
		t3 = other.t3;
		e1 = other.e1;
		e2 = other.e2;
	}

	__host__ __device__ Triangle(const int& id, const Vec4& pt1, const Vec4& pt2, const Vec4& pt3, const int& MatID) :
		id(id),
		MatID(MatID)
	{
		isSmooth = false;
		hastxt = false;

		p1 = pt1;
		p2 = pt2;
		p3 = pt3;
		e1 = p2 - p1;
		e2 = p3 - p1;
		normal = cross(e2, e1);
		normal.norm();
	}

	__host__ __device__ Triangle(const int& id, const Vec4& pt1, const Vec4& pt2, const Vec4& pt3, const Vec4& nv1, const Vec4& nv2, const Vec4& nv3, const int& MatID) :
		id(id),
		MatID(MatID)
	{
		isSmooth = true;
		hastxt = false;
		p1 = pt1;
		p2 = pt2;
		p3 = pt3;
		n1 = nv1;
		n2 = nv2;
		n3 = nv3;
		e1 = p2 - p1;
		e2 = p3 - p1;
	}

	__host__ __device__ Triangle(const int& id, const Vec4& pt1, const Vec4& pt2, const Vec4& pt3, const Vec4& nv1, const Vec4& nv2, const Vec4& nv3, const Vec4& tv1, const Vec4& tv2, const Vec4& tv3, const int& MatID) :
		id(id),
		MatID(MatID)
	{
		isSmooth = true;
		hastxt = true;
		p1 = pt1;
		p2 = pt2;
		p3 = pt3;
		n1 = nv1;
		n2 = nv2;
		n3 = nv3;
		t1 = tv1;
		t2 = tv2;
		t3 = tv3;
		e1 = p2 - p1;
		e2 = p3 - p1;
	}

	__host__ __device__ void transform(const Mat4& m) {
		p1 = m * p1;
		p2 = m * p2;
		p3 = m * p3;
		e1 = p2 - p1;
		e2 = p3 - p1;

		if (!isSmooth) {
			normal = cross(e2, e1);
			normal.norm();
		}
		else {
			n1 = m * n1;
			n2 = m * n2;
			n3 = m * n3;
			n1.norm();
			n2.norm();
			n3.norm();
		}
	}


	/*
	* Normal
	*/
	__host__ __device__ Vec4 normal_at(const Vec4& v, const float& uIN, const float& vIN) {

		if (!isSmooth)
			return normal;

		Vec4 calc_n = n2 * uIN + n3 * vIN + n1 * (1 - uIN - vIN);
		return calc_n;
	}

	__host__ __device__ Vec4 texture_at(const Vec4& v, const float& uIN, const float& vIN) {


		Vec4 ts1(t1);
		Vec4 ts2(t2);
		Vec4 ts3(t3);

		Vec4 calc_n = t2 * uIN + t3 * vIN + t1 * (1 - uIN - vIN);
		return calc_n;
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


		//printf("Check intersection\n");

		Ray cpy = ray;
		Vec4 rayDir = cpy.getDirection();
		Vec4 rayOri = cpy.getOrigin();

		Vec4 dir_cross_e2 = cross(rayDir, e2);
		float det = dot(e1, dir_cross_e2);

		if (std::abs(det) < EPSILON)
			return;

		float f = 1.0f / det;
		Vec4 p1_to_origin = rayOri - p1;

		float u = f * dot(p1_to_origin, dir_cross_e2);

		if (u < 0 || u > 1)
			return;

		Vec4 origin_cross_e1 = cross(p1_to_origin, e1);
		float v = f * dot(rayDir, origin_cross_e1);

		if (v < 0 || (u + v) > 1)
			return;
		float t = f * dot(e2, origin_cross_e1);

		i.push(id, t, this, u, v);
		//i.observations.emplace_back(id, t, 5, u, v);

	}

	__host__ __device__ Triangle& operator =(const Triangle& t) {

		id = t.id;
		MatID = t.MatID;
		p1 = t.p1;
		p2 = t.p2;
		p3 = t.p3;
		e1 = t.e1;
		e2 = t.e2;
		isSmooth = t.isSmooth;
		hastxt = t.hastxt;

		if (isSmooth) {
			n1 = t.n1;
			n2 = t.n2;
			n3 = t.n3;
		}
		else {
			normal = t.normal;
		}
		if (hastxt) {
			t1 = t.t1;
			t2 = t.t2;
			t3 = t.t3;
		}

		return *this;
	}

	__host__ __device__ Triangle& operator =(Triangle&& t) {

		id = t.id;
		MatID = t.MatID;
		p1 = t.p1;
		p2 = t.p2;
		p3 = t.p3;
		e1 = t.e1;
		e2 = t.e2;
		isSmooth = t.isSmooth;
		hastxt = t.hastxt;

		if (isSmooth) {
			n1 = t.n1;
			n2 = t.n2;
			n3 = t.n3;
		}
		else {
			normal = t.normal;
		}
		if (hastxt) {
			t1 = t.t1;
			t2 = t.t2;
			t3 = t.t3;
		}

		return *this;
	}

	__host__ __device__ float getMinX() {
		return p1.x < p2.x ? (p1.x < p3.x ? p1.x : p3.x) : (p2.x < p3.x ? p2.x : p3.x);
	}
	__host__ __device__ float getMinY() {
		return p1.y < p2.y ? (p1.y < p3.y ? p1.y : p3.y) : (p2.y < p3.y ? p2.y : p3.y);
	}
	__host__ __device__ float getMinZ() {
		return p1.z < p2.z ? (p1.z < p3.z ? p1.z : p3.z) : (p2.z < p3.z ? p2.z : p3.z);
	}

	__host__ __device__ float getMaxX() {
		return p1.x > p2.x ? (p1.x > p3.x ? p1.x : p3.x) : (p2.x > p3.x ? p2.x : p3.x);
	}
	__host__ __device__ float getMaxY() {
		return p1.y > p2.y ? (p1.y > p3.y ? p1.y : p3.y) : (p2.y > p3.y ? p2.y : p3.y);
	}
	__host__ __device__ float getMaxZ() {
		return p1.z > p2.z ? (p1.z > p3.z ? p1.z : p3.z) : (p2.z > p3.z ? p2.z : p3.z);
	}
};



class TriangleMesh {
private:


	//Mat4 transformation;

public:

	CUDAVector<Triangle> triangles;
	int numT;
	int start;
	int end;

	int MatID;
	Cube BoundingBox;


	/*
	* Constructor
	*/
	__host__ __device__ TriangleMesh(CUDAVector<Triangle>& trianglesIN, const int& start, const int& end) :
		//triangles(trianglesIN),
		start(start),
		end(end)
	{
		triangles = trianglesIN;
		numT = end - start;
		createBB();
	}

	__host__ __device__ void intersect(Intersection& i, const Ray& ray) {

		if( BoundingBox.intersect(ray) ) {
			for (int k = start; k < end; ++k) {
				triangles[k].intersect(i, ray);
			}
		}
	}

	/*
	* MATERIAL
	*/
	__host__ __device__ int getMatID() {
		return MatID;
	}

	__host__ __device__ void createBB() {

		std::pair<float, float> x_minmax = { INFINITY, -INFINITY };
		std::pair<float, float> y_minmax = { INFINITY, -INFINITY };
		std::pair<float, float> z_minmax = { INFINITY, -INFINITY };

		for (int i = start; i < end; i++) {

			float minx, miny, minz;
			float maxx, maxy, maxz;
			minx = triangles[i].getMinX();
			miny = triangles[i].getMinY();
			minz = triangles[i].getMinZ();
			maxx = triangles[i].getMaxX();
			maxy = triangles[i].getMaxY();
			maxz = triangles[i].getMaxZ();

			if (minx < x_minmax.first) x_minmax.first = minx;
			if (miny < y_minmax.first) y_minmax.first = miny;
			if (minz < z_minmax.first) z_minmax.first = minz;

			if (maxx > x_minmax.second) x_minmax.second = maxx;
			if (maxy > y_minmax.second) y_minmax.second = maxy;
			if (maxz > z_minmax.second) z_minmax.second = maxz;
		}

		float delta_x = (x_minmax.second - x_minmax.first) / 2;
		float delta_y = (y_minmax.second - y_minmax.first) / 2;
		float delta_z = (z_minmax.second - z_minmax.first) / 2;

		delta_x = delta_x <= 0.0f ? 0.1f : delta_x;
		delta_y = delta_y <= 0.0f ? 0.1f : delta_y;
		delta_z = delta_z <= 0.0f ? 0.1f : delta_z;

		Mat4 t = transform(
			delta_x + x_minmax.first,
			delta_y + y_minmax.first,
			delta_z + z_minmax.first,
			delta_x, delta_y, delta_z
		);

		BoundingBox = Cube(-1, t, -1);
	}
};

#endif