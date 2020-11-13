#ifndef WORLD_CUH
#define WORLD_CUH

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// local includes
#include "Vec4.hpp"
#include "Color.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "Comps.hpp"

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Cube.hpp"
#include "Triangle.hpp"

#include "OBJ_READER.hpp"

#include "CUDAVector.cuh"
#include "Texture.hpp"


class World {
public:

    __host__ void print() {
        std::cerr << "NumMaterials: " << materials_vector.size() << std::endl;
        std::cerr << "NumTextures: " << textures_vector.size() << std::endl;
        std::cerr << "NumLights: " << lights_vector.size() << std::endl;
        std::cerr << "NumSpheres: " << spheres_vector.size() << std::endl;
        std::cerr << "NumPlanes: " << planes_vector.size() << std::endl;
        std::cerr << "NumCubes: " << cubes_vector.size() << std::endl;
        std::cerr << "NumTriangles: " << triangle_vector.size() << "\nNumGroups: " << mesh_vector.size() << "\n";
    }

    __device__ void dprintf() {
        printf("NumMaterials: %d\n", materials_vector.size());
        printf("NumTextures: %d\n", textures_vector.size());
        printf("NumLights: %d\n", lights_vector.size());
        printf("NumTriangles: %d\n", triangle_vector.size());
        printf("NumGroups: %d\n", mesh_vector.size());
    }

    // Camera
    Camera* camera;

    /*
    Constructor
    */
    __host__ World() {
    }

    __host__ void createWorld(const std::string& filename);

    /*
    Destructor - free all pointers
    */
    __host__ ~World() {
        checkCudaErrors(cudaFree(camera));
    }

    __host__ __device__ int getWidth() {
        return width;
    }

    __host__ int getHeight() {
        return height;
    }


    __host__ __device__ void intersect(Intersection& intersection, const Ray& ray) {


        int x = triangle_vector.size();

        for (int i = 0; i < spheres_vector.size(); i++) {
            spheres_vector[i].intersect(intersection, ray);
        }
        for (int i = 0; i < planes_vector.size(); i++) {
            planes_vector[i].intersect(intersection, ray);
        }
        for (int i = 0; i < cubes_vector.size(); i++) {
            cubes_vector[i].intersect(intersection, ray);
        }
        for (int i = 0; i < mesh_vector.size(); i++) {
            mesh_vector[i].intersect(intersection, ray);
        } 
        
    }

    /*
    * COLOR FUNCTIONS
    */

    __host__ __device__ void checkShade(bool& isShade, const Vec4& pt, const Light& l) {

        Vec4 v = l.position - pt;

        float distance = v.length();
        v.norm();

        Vec4 offset = v * 0.5f + pt;

        Ray ray(offset, v);
        Intersection intersection;

        intersect(intersection, ray);

        if (intersection.goodTs() > 0 && intersection.getT() < distance) {
            isShade = true;
        }
        else {
            isShade = false;
        }
    }

    __host__ __device__ void checkShade2(float& shadeIntensity, const Vec4& pt, const Light& lc) {

        Light l = lc;
        int numPts = l.getNumPts();
        int hits = 0;

        for (int i = 0; i < numPts; ++i) {
            
            Vec4 lpt = l.getPt(i);
            Vec4 dir = lpt - pt;
            dir.w = 0.0f;
            float dist = dir.length();
            dir.norm();

            Ray r(pt, dir);
            Intersection in;
            intersect(in, r);

            if (in.goodTs() > 0 && in.getT() < dist)
                continue;
            ++hits;
        }
        shadeIntensity = (float)hits / (float)numPts;
    }


    __host__ __device__ void lighting(const Material& material, const Light& light, const Vec4& pt, const Vec4& eyeVector, const Vec4& normalVector, Color& color) {

        Color effectiveColor(0), ambientColor(0), diffuseColor(0), specularColor(0);

        Vec4 lightPosition = light.position;
        Vec4 lightVec = lightPosition - pt;
        lightVec.norm();

        //effectiveColor = material.color * light.intensity;
        ambientColor = material.ambient;
        

        float lightDotNormal = dot(lightVec, normalVector);

        bool isShade = false;
        checkShade(isShade, pt, light);

        // diffuse shade
        float shadeIntensity = 1.0f;

        if (isShade) {
            color = ambientColor;
            return;
        }

        /*
        if (!isShade) {

            color = ambientColor;
            return;

            /*
            float shadeIntensity = 0.0f;
            Vec4 ptOff = pt + 0.1f * normalVector;
            checkShade2(shadeIntensity, ptOff, light);

            if (shadeIntensity == 0.0f) {
                color = ambientColor;
                return;
            }
            
        }
        */

        if (lightDotNormal > 0) {
            diffuseColor = material.diffuse * lightDotNormal;
            lightVec *= -1;
            Vec4 reflectVector = reflect(lightVec, normalVector);
            float reflectDotEye = dot(reflectVector, eyeVector);
            if (reflectDotEye > 0) {
                float factor = std::pow(reflectDotEye, material.shinyness);
                specularColor = light.intensity * material.specular * factor;
            }
        }
        color = ambientColor + diffuseColor + specularColor;
    }

    // TODO: implement reflection / refraction

    __host__ __device__ void color_at(const Comps& comps, Color& color) {

        for (int i = 0; i < lights_vector.size(); ++i) {

            Color tmp(0);
            Light localLight = lights_vector[i];
            Material localMat = materials_vector[comps.MatID];
    
            lighting(localMat, localLight, comps.point, comps.eye, comps.normal, tmp);
            

            color += tmp;
        }
        //return;
        color.clamp();
    }

    __host__ __device__ void color_at2(const Comps& comps, Color& color) {

        for (int i = 0; i < lights_vector.size(); ++i) {
            Color tmp(0);
            Light localLight = lights_vector[i];

            lighting(comps.material, localLight, comps.point, comps.eye, comps.normal, tmp);

            color += tmp;
        }
        //return;
        color.clamp();
    }
    

    __host__ __device__ void color_at3(const Comps& comps, Image_Buffer& image_buffer) {

        for (int i = 0; i < lights_vector.size(); ++i) {
            Color tmp(0);
            Light localLight = lights_vector[i];

            lighting(comps.material, localLight, comps.point, comps.eye, comps.normal, tmp);

            if (comps.material.reflective > 0.0f) {
                image_buffer.reflects = true;
                image_buffer.reflection_ray = Ray(comps.over_point, comps.reflectv);
            }

            if (comps.material.transparent < 1.0f) {
                
                // calculate refraction direction
                Vec4 newDirection = comps.eye * -1;

                // current n value
                float n1 = image_buffer.refractive_index;
                float n2 = comps.material.refractive_index;

                // compute 
                float n_ratio = n1/n2;
                float cos_i = dot(comps.eye, comps.normal);
                float sin2_t = n_ratio * n_ratio * (1 - cos_i * cos_i);

                if (sin2_t > 1.0f) continue; // total internal reflection

                float cos_t = std::sqrt(1.0f - sin2_t);

                Vec4 direction = comps.normal * (n_ratio * cos_i - cos_t) - comps.eye * n_ratio;

                image_buffer.refracts = true;
                image_buffer.refraction_ray = Ray(comps.under_point, direction);
                image_buffer.refractive_index = n2;
            }

            // decrement the visual intensity of refraction
            image_buffer.intensity -= comps.material.transparent;
            // seen material * (reflection strength - 1) * transparency
            color += (tmp * (1 - comps.material.reflective)) * comps.material.transparent;
        }
        color.clamp();
    }

    __host__ __device__ Comps prepare_computations(Intersection& intersection, Ray& ray);

    /*
    * New Vector class
    */
    CUDAVector<Sphere> spheres_vector;
    CUDAVector<Plane> planes_vector;
    CUDAVector<Cube> cubes_vector;
    CUDAVector<TriangleMesh> mesh_vector;

    CUDAVector<Triangle> triangle_vector;
    CUDAVector<Material> materials_vector;
    CUDAVector<Texture> textures_vector;

    CUDAVector<Light> lights_vector;

private:

    /*
    * General
    */

    int width, height;
    float aspectRatio;





};


__host__ __device__ Comps World::prepare_computations(Intersection& intersection, Ray& ray) {
    
    Comps comps;

    Observation* obs = intersection.getObs(0);

    // fill intersection values
    comps.t = obs->t;
    comps.point = ray.position(comps.t);

    // get shape normal
    int shapetype = obs->shapetype;
    int shapeid obs->shapeid;
    int matid = 0;

    if (shapetype == 1) {
        comps.normal = shapes_vector[shapeid].normal_at(comps.point);
        matid = shapes_vector[shapeid].getMatID();
    }
    else if (shapetype == 2) {
        comps.normal = planes_vector[shapeid].normal_at(comps.point);
        matid = planes_vector[shapeid].getMatID();
    }
    else if (shapetype == 3) {
        comps.normal = cubes_vector[shapeid].normal_at(comps.point);
        matid = cubes_vector[shapeid].getMatID();
    }
    else if (shapetype == 4) {
        comps.normal = cylinders_vector[shapeid].normal_at(comps.point);
        matid = cylinders_vector[shapeid].getMatID();
    }
    else if (shapetype == 5) {
        comps.normal = triangle_vector[shapeid].normal_at(comps.point, obs->u, obs->v);
        matid = triangle_vector[shapeid].getMatID();
    }

    Material tmp(materials_vector[matid]);
    int txt = tmp.textureid;

    // if texture is unequal to -1 it has been set and needs to be mapped
    if (txt != -1) {

        Vec4 txtUV = triangle_vector[idx].texture_at(comps.point, obs->u, obs->v);

        // map ambient
        if (tmp.mapA == 1) {
            tmp.ambient = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        // map diffuse
        if (tmp.mapD == 1) {
            tmp.ambient = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        // map ambient
        if (tmp.mapS == 1) {
            tmp.specular = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        if (tmp.mapTransparent == 1) {
            tmp.transparent = textures_vector[txt].mapD(txtUV.x, txtUV.y);
        }
    }
    
    

    comps.material = tmp;

    comps.reflectv = reflect(ray.getDirection(), comps.normal);
    comps.eye = ray.getDirection() * -1;
    
    // new
    comps.under_point = comps.point + EPSILON * ray.getDirection();
    comps.over_point = comps.point + EPSILON * comps.reflectv;
    //
    
    float nDotE = dot(comps.normal, comps.eye);
    if (nDotE < 0) {
        comps.inside = true;
        comps.normal = comps.normal * -1;
    }
    else {
        comps.inside = false;
    }

    return comps;
}

__host__ __device__ Comps World::prepare_computations(Intersection& intersection, Ray& ray) {
    
    Comps comps;

    // fill intersection values
    comps.t = intersection.getT();
    comps.point = ray.position(comps.t);

    // get shape normal
    int idx = intersection.getTriangleId();
    comps.normal = triangle_vector[idx].normal_at(comps.point, intersection.getU(), intersection.getV());

    // get shape material and fill new
    int mat = triangle_vector[idx].getMatID();
    Material tmp(materials_vector[mat]);
    int txt = tmp.textureid;

    // if texture is unequal to -1 it has been set and needs to be mapped
    if (txt != -1) {

        Triangle tritest(triangle_vector[idx]);

        Vec4 txtUV = triangle_vector[idx].texture_at(comps.point, intersection.getU(), intersection.getV());

        // map ambient
        if (tmp.mapA == 1) {
            tmp.ambient = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        // map diffuse
        if (tmp.mapD == 1) {
            tmp.ambient = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        // map ambient
        if (tmp.mapS == 1) {
            tmp.specular = textures_vector[txt].mapColor(txtUV.x, txtUV.y);
        }

        if (tmp.mapTransparent == 1) {
            //tmp.color = Color(0);
        }
    }

    comps.material = tmp;


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

    return comps;
}


__host__ void World::createWorld(const std::string& filename) {

    std::ifstream file_(filename);
    if (!file_.is_open()) {
        printf("World file not found!");
        return;
    }

    // struct to retrieve input
    struct datareader {
        int id, matID;
        Mat4 m;
        datareader(std::stringstream& input) {
            float x_translate, y_translate, z_translate, x_scale, y_scale, z_scale, x_rotate, y_rotate, z_rotate, xy, xz, yx, yz, zx, zy;
            input >> id >> matID >> x_translate >> y_translate >> z_translate >> x_scale >> y_scale >> z_scale >> x_rotate >> y_rotate >> z_rotate >> xy >> xz >> yx >> yz >> zx >> zy;
            m = transform(x_translate, y_translate, z_translate, x_scale, y_scale, z_scale, x_rotate, y_rotate, z_rotate, xy, xz, yx, yz, zx, zy);
        }
    };

    std::string line_;
    while (getline(file_, line_)) {
        if (line_[0] == '#') continue;
        if (line_.empty()) continue;

        std::stringstream input(line_);
        std::string paramName;

        input >> paramName;

        if (paramName == "Pixel:") {
            input >> width >> height;
            aspectRatio = (float)width / height;
        }


        /*
        * CAMERA
        */
        else if (paramName == "Camera:") {
            float px, py, pz, fx, fy, fz, ux, uy, uz, fov;
            input >> px >> py >> pz >> fx >> fy >> fz >> ux >> uy >> uz >> fov;

            Vec4 origin(px, py, pz, 1.0f);
            Vec4 forward(fx, fy, fz);
            Vec4 upguide(ux, uy, uz);
            fov = fov * PI / 180.f;
            checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
            camera->createCamera(origin, forward, upguide, fov, aspectRatio);
        }

        /*
        * MATERIAL
        */
        else if (paramName == "Material") {
            int id;
            float r, g, b, ambient, diffuse, specular, shinyness;
            input >> id >> r >> g >> b >> ambient >> diffuse >> specular >> shinyness;
            Color tmp(r, g, b);
            materials_vector.emplace_back(tmp, ambient, diffuse, specular, shinyness);
        }

        /*
        * SPHERE
        */
        else if (paramName == "Sphere") {
            datareader d(input);
            spheres_vector.emplace_back(spheres_vector.size(), d.m, d.matID);
        }

        /*
        * PLANE
        */
        else if (paramName == "Plane") {
            datareader d(input);
            planes_vector.emplace_back(planes_vector.size(), d.m, d.matID);
        }

        /*
        * CUBE
        */
        else if (paramName == "Cube") {
            datareader d(input);
            cubes_vector.emplace_back(cubes_vector.size(), d.m, d.matID);
        }

        /*
        * OBJ File
        */
        else if (paramName == "OBJ") {

            datareader d(input);
            std::string filename;
            input >> filename;

            OBJ_Reader o(filename, triangle_vector, materials_vector, textures_vector);

            //mesh_vector.reserve(o.getNumGroups() + mesh_vector.size());
            for (int i = 0; i < o.getNumGroups(); i++) {
                std::pair<int, int> bounds = o.getGroupBounds(i);
                mesh_vector.emplace_back(triangle_vector, bounds.first, bounds.second);
            }
            int test1 = o.getNumGroups();
            int test = 0;
        }

        /*
        * LIGHT
        */
        else if (paramName == "Light") {
            int id;
            float px, py, pz, r, g, b;
            input >> id >> px >> py >> pz >> r >> g >> b;
            lights_vector.emplace_back(Color(r, g, b), Vec4(px, py, pz, 1.0f));
        }

        else {
            std::cerr << "***Unrecognized input*** -> " << paramName << std::endl;
        }
    }

    file_.close();
    return;
}

#endif
