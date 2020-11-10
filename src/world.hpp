ifndef WORLD_HPP
#define WORLD_HPP

#include <string>
#include <fstream>
#include <sstream>

#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "hittable.hpp"

#include "Ray.hpp"
#include "Intersection.hpp"


struct Comps {
	float t;
	hittable* object;
	Vec4 point;
	Vec4 eye;
	Vec4 normal;
	bool inside;
};

class World {

private:

	int width, height;
	float aspectRatio;

	hittable** objects;
	Light* lights;
	Material* materials;

	int numObj;
	int numLights;
	int numMats;



public:

	Camera* camera;

	World() {};
	__host__ ~World() {
		delete[] camera;
		delete[] objects;
		delete[] lights;
		delete[] materials;
	}

	__host__ __device__ int getWidth() {
		return width;
	}

	__host__ int getHeight() {
		return height;
	}


	// __global__ 
	__host__ void createWorld(const std::string& filename) {
		std::ifstream file_(filename);
		if (!file_.is_open()) {
			printf("World file not found!");
			return;
		}
		printf("Reading file!");
		//std::cout << "Reading File: " << filename << std::endl;

		std::string line_;
		//int i = 0;
		while (getline(file_, line_)) {
			if (line_[0] == '#') continue;
			if (line_.empty()) continue;

			std::stringstream input(line_);
			std::string paramName;

			input >> paramName;
			// read scene

			if (paramName == "Pixel:") {
				input >> width >> height;
				aspectRatio = (float)width / height;

				printf("Rendering a world with %dx%d pixels, aspect ratio %f\n", width, height, aspectRatio);

			}
			else if (paramName == "Camera:") {
				float px, py, pz, fx, fy, fz, ux, uy, uz, fov;
				input >> px >> py >> pz >> fx >> fy >> fz >> ux >> uy >> uz >> fov;

				Vec4 origin(px, py, pz);
				Vec4 forward(fx, fy, fz);
				Vec4 upguide(ux, uy, uz);
				fov = fov * PI / 180.f;

				// has to be allocated in GPU memory

				//World* world;
				checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(World*)));
				world->createWorld("Scene.txt");

				camera = new Camera(origin, forward, upguide, fov, aspectRatio);

				printf("Created a camera\n");
			}
			else if (paramName == "NumObjects:") {
				//int numObj;
				input >> numObj;
				objects = new hittable * [numObj];
				printf("Reserving space for %d objects\n", numObj);
				numObj = 0;
			}
			else if (paramName == "NumLights:") {
				//int numLights;
				input >> numLights;
				lights = new Light[numLights];
				printf("Reserving space for %d lights\n", numLights);
				numLights = 0;
			}
			else if (paramName == "NumMaterials:") {
				//int numMats;
				input >> numMats;
				materials = new Material[numMats];
				printf("Reserving space for %d materials\n", numMats);
				numMats = 0;
			}
			else if (paramName == "Material") {
				int id;
				float r, g, b, ambient, diffuse, specular, shinyness;
				input >> id >> r >> g >> b >> ambient >> diffuse >> specular >> shinyness;
				Color tmp(r, g, b);

				materials[id] = Material(tmp, ambient, diffuse, specular, shinyness);
				printf("Loading mat %d\n", id);
				numMats++;
			}
			else if (paramName == "Sphere") {
				int id, matID;
				float x_translate, y_translate, z_translate, x_scale, y_scale, z_scale, x_rotate, y_rotate, z_rotate, xy, xz, yx, yz, zx, zy;
				input >> id >> x_translate >> y_translate >> z_translate >> x_scale >> y_scale >> z_scale >> x_rotate >> y_rotate >> z_rotate >> xy >> xz >> yx >> yz >> zx >> zy >> matID;
				Mat4 m = transform(x_translate, y_translate, z_translate, x_scale, y_scale, z_scale, x_rotate, y_rotate, z_rotate, xy, xz, yx, yz, zx, zy);
				hittable* s1 = new Sphere(id, m, materials[matID]);
				objects[id] = s1;
				printf("Loading shape %d\n", id);
				numObj++;
			}
			else if (paramName == "Light") {
				int id;
				float px, py, pz, r, g, b;
				input >> id >> px >> py >> pz >> r >> g >> b;
				Color tmp(r, g, b);
				Vec4 pos(px, py, pz);
				lights[id] = new Light(tmp, pos);
				printf("Loading light %d\n", id);
				numLights++;

			}
			else {
				printf("***Unrecognized input***\n");
			}
		}

		file_.close();
		return;
	}
};



#endif