#ifndef COMPS_HPP
#define COMPS_HPP

#include "Vec4.hpp"
#include "Material.hpp"

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Cube.hpp"
#include "Cylinder.hpp"
#include "Triangle.hpp"

struct Comps {

	float t;
	Vec4 point;
	Vec4 eye;
	Vec4 normal;
	Vec4 reflectv;
	bool inside;
	int MatID;

	Material material;

};

__host__ __device__ void fillComps(Intersection& intersection, Ray& ray, Comps& comps) {

	comps.reflectv = reflect(ray.getDirection(), comps.normal);
	comps.eye = ray.getDirection() * -1;
	float nDotE = dot(comps.normal, comps.eye);
	if (nDotE < 0) {
		comps.inside = true;
		comps.normal = comps.normal * -1;
	}
	else {
		comps.inside = false;
	}
}
// new
__host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray, Sphere& s) {

	Comps comps;
	comps.t = intersection.getT();
	comps.point = ray.position(comps.t);
	comps.normal = s.normal_at(comps.point);
	comps.MatID = s.getMatID();

	fillComps(intersection, ray, comps);
	return comps;
}

__host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray, Plane& p) {

	Comps comps;
	comps.t = intersection.getT();
	comps.point = ray.position(comps.t);
	comps.normal = p.normal_at(comps.point);
	comps.MatID = p.getMatID();

	fillComps(intersection, ray, comps);
	return comps;
}

__host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray, Cube& c) {

	Comps comps;
	comps.t = intersection.getT();
	comps.point = ray.position(comps.t);
	comps.normal = c.normal_at(comps.point);
	comps.MatID = c.getMatID();

	fillComps(intersection, ray, comps);
	return comps;
}

__host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray, Cylinder& c) {

	Comps comps;
	comps.t = intersection.getT();
	comps.point = ray.position(comps.t);
	comps.normal = c.normal_at(comps.point);
	comps.MatID = c.getMatID();

	fillComps(intersection, ray, comps);
	return comps;
}

__host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray, Triangle& t) {

	Comps comps;
	comps.t = intersection.getT();
	comps.point = ray.position(comps.t);
	comps.normal = t.normal_at(comps.point, intersection.getU(), intersection.getV());
	comps.MatID = t.getMatID();

	fillComps(intersection, ray, comps);
	return comps;
}

#endif