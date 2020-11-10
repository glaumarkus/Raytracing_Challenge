#ifndef INTERSECTION_HPP
#define INTERSECTION_HPP

#include "Vec4.hpp"
#include "Ray.hpp"

#include "CUDAVector.cuh"

//class hittable;
class Sphere;
class Plane;
class Cube;
class Cylinder;
class Triangle;

class Observation {
private:
public:

    int shapeId, id;
    float t, u, v;

    __host__ __device__ Observation() {}

    __host__ __device__ Observation(const int& id, const int& shapeId, const float& t) :
        shapeId(shapeId),
        id(id),
        t(t),
        u(INFINITY),
        v(INFINITY)
    {}

    __host__ __device__ Observation(const int& id, const int& shapeId, const float& t, const float& u, const float& v) :
        shapeId(shapeId),
        id(id),
        t(t),
        u(u),
        v(v)
    {}

};


class Intersection {
private:

    //Observation* obs;

    int* id;
    float* t;

    int capacity;
    int current;
    int goodT;
    float bestT;


    // Shape primitves
    bool spherePTR;
    int sphereId;

    bool planePTR;
    int planeId;

    bool cubePTR;
    int cubeId;

    bool cylinderPTR;
    int cylinderId;

    bool trianglePTR;
    int triangleId;
    float u, v;


    // set new hit from id
    __host__ __device__ void setHit(const int& newid, const int& newShape) {

        switch (newShape) {
        case 1: {
            spherePTR = true;
            sphereId = newid;
        }
        case 2: {
            planePTR = true;
            planeId = newid;
        }
        case 3: {
            cubePTR = true;
            cubeId = newid;
        }
        case 4: {
            cylinderPTR = true;
            cylinderId = newid;
        }
        case 5: {
            trianglePTR = true;
            triangleId = newid;
        }
        default: {}
        }
    }

    __host__ __device__ void pushINT(const int& newid, const float& newt, const int& newShape, const float& uIN, const float& vIN) {
        /*
        Mapping shapes:
        1 = Sphere
        2 = Plane
        3 = Cube
        4 = Cylinder
        5 = Triangle
        */

        if (current == capacity) {

            int* temp_id = new int[2 * capacity];
            float* temp_t = new float[2 * capacity];

            for (int i = 0; i < capacity; ++i) {
                temp_id[i] = id[i];
                temp_t[i] = t[i];
            }

            delete[] id;
            delete[] t;

            capacity *= 2;
            id = temp_id;
            t = temp_t;
        }

        id[current] = newid;
        t[current] = newt;

        current++;

        if (newt > 0) {
            goodT++;
            if (newt < bestT && newt > 0.0f) {
                resetBools();
                bestT = newt;
                setHit(newid, newShape);

                // use u & v with triangles
                if (newShape == 5) {
                    u = uIN;
                    v = vIN;
                }
            }
        }
    }


    __host__ __device__ void resetBools() {
        spherePTR = false;
        planePTR = false;
        cylinderPTR = false;
        cubePTR = false;
        trianglePTR = false;
        u = INFINITY;
        v = INFINITY;
    }


public:

    __device__ void print() {
        for (int i = 0; i < current; i++) {
            printf("Observation: %d with t: %f\n", i, t[i]);
        }
        //if (observations.size() == 0) printf("Intersection is empty\n");
    }

    __host__ __device__ Intersection()
    {
        id = new int[10];
        t = new float[10];
        capacity = 10;
        current = 0;
        goodT = 0;
        bestT = INFINITY;
        resetBools();
    }

    __host__ __device__ ~Intersection() {
        delete[] id;
        delete[] t;
    }


    __host__ __device__ int goodTs() {
        return goodT;
    }

    __host__ __device__ float getT() {
        return bestT;
    }



    // individual pushes for each primitive
    __host__ __device__ void push(const int& newid, const float& newt, Sphere* sphereHit) {
        pushINT(newid, newt, 1, INFINITY, INFINITY);
    }

    __host__ __device__ void push(const int& newid, const float& newt, Plane* planeHit) {
        pushINT(newid, newt, 2, INFINITY, INFINITY);
    }

    __host__ __device__ void push(const int& newid, const float& newt, Cube* cubeHit) {
        pushINT(newid, newt, 3, INFINITY, INFINITY);
    }

    __host__ __device__ void push(const int& newid, const float& newt, Cylinder* cylinderHit) {
        pushINT(newid, newt, 4, INFINITY, INFINITY);
    }

    __host__ __device__ void push(const int& newid, const float& newt, Triangle* triangleHit, const float& uIN, const float& vIN) {
        pushINT(newid, newt, 5, uIN, vIN);
    }




    // shape primitive functions -- HIT
    __host__ __device__ bool hitIsSphere() {
        return spherePTR;
    }
    __host__ __device__ bool hitIsPlane() {
        return planePTR;
    }
    __host__ __device__ bool hitIsCube() {
        return cubePTR;
    }
    __host__ __device__ bool hitIsCylinder() {
        return cylinderPTR;
    }
    __host__ __device__ bool hitIsTriangle() {
        return trianglePTR;
    }

    // shape primitive functions -- get ID
    __host__ __device__ int getSphereId() {
        return sphereId;
    }
    __host__ __device__ int getPlaneId() {
        return planeId;
    }
    __host__ __device__ int getCubeId() {
        return cubeId;
    }
    __host__ __device__ int getCylinderId() {
        return cylinderId;
    }
    __host__ __device__ int getTriangleId() {
        return triangleId;
    }

    // triangle specific
    __host__ __device__ float getU() const {
        return u;
    }
    __host__ __device__ float getV() const {
        return v;
    }


};






#endif