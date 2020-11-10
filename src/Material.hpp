#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include <string>
#include "Color.hpp"

class Material {
public:

    //std::string name;
    int id;

    Color color;
    Color ambient;
    Color diffuse;
    Color specular;

    float shinyness;
    float reflective;
    float transparent;
    float refractive;

    int textureid = -1;

    int mapA = 0;
    int mapD = 0;
    int mapS = 0;
    int mapTransparent = 0;

    /*
    Vacuum: 1
    Air: 1.00029
    Water: 1.333
    Glass: 1.52
    Diamond: 2.417
    */

    __host__ void print() {
        printf("Material %d\n", id);
    }

    __host__ __device__ Material() {
        color = Color();
        ambient = Color(0.1f);
        diffuse = Color(0.9f);
        specular = Color(0.9f);
        shinyness = 20.0f;
        reflective = 0.0f;
        transparent = 0.0f;
        refractive = 1.0f;
       
    };

    __host__ __device__ Material& operator =(const Material& other) {
        color = other.color;
        ambient = other.ambient;
        diffuse = other.diffuse;
        specular = other.specular;
        shinyness = other.shinyness;
        reflective = other.reflective;
        transparent = other.transparent;
        refractive = other.refractive;

        textureid = other.textureid;
        mapA = other.mapA;
        mapD = other.mapD;
        mapS = other.mapS;
        mapTransparent = other.mapTransparent;

        return *this;
    }

    __host__ __device__ Material& operator =(const Material&& other) {
        color = other.color;
        ambient = other.ambient;
        diffuse = other.diffuse;
        specular = other.specular;
        shinyness = other.shinyness;
        reflective = other.reflective;
        transparent = other.transparent;
        refractive = other.refractive;

        textureid = other.textureid;
        mapA = other.mapA;
        mapD = other.mapD;
        mapS = other.mapS;
        mapTransparent = other.mapTransparent;

        return *this;
    }

    __host__ __device__ Material(
        const Color& color,
        const float& ambientF,
        const float& diffuseF,
        const float& specularF,
        const float& shinyness) :
        color(color),
        shinyness(shinyness)
    {
        ambient = Color(ambientF);
        diffuse = Color(diffuseF);
        specular = Color(specularF);
    }

    __host__ __device__ Material (const Material& other) {
        color = other.color;
        ambient = other.ambient;
        diffuse = other.diffuse;
        specular = other.specular;
        shinyness = other.shinyness;
        reflective = other.reflective;
        transparent = other.transparent;
        refractive = other.refractive;
        
        textureid = other.textureid;
        mapA = other.mapA;
        mapD = other.mapD;
        mapS = other.mapS;
        mapTransparent = other.mapTransparent;
    }

    __host__ void initMat(
        const Color& colorIN,
        const float& ambientIN,
        const float& diffuseIN,
        const float& specularIN,
        const float& shinynessIN
    ) {
        color = colorIN;
        ambient = Color(ambientIN);
        diffuse = Color(diffuseIN);
        specular = Color(specularIN);
        shinyness = shinynessIN;
    }

    __host__ void initMat(
        const Color& colorIN,
        const Color& ambientIN,
        const Color& diffuseIN,
        const Color& specularIN,
        const float& shinynessIN
    ) {
        color = colorIN;
        ambient = Color(ambientIN);
        diffuse = Color(diffuseIN);
        specular = Color(specularIN);
        shinyness = shinynessIN;
    }

};
#endif 