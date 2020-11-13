#ifndef IMAGE_BUFFER_HPP
#define IMAGE_BUFFER_HPP

#include "Color.hpp"
#include "Ray.hpp"


class Image_buffer {
public:

    __host__ image_buffer() {}

    __device__ void update(const Color& color, const float& color_intensity) {
        pixel_color += (color * color_intensity);
        intensity -= color_intensity;
    }

    __device__ Ray& getReflection() {
        return reflection_ray;
    }
    __device__ Ray& getRefraction() {
        return refraction_ray;
    }

        // Ray params
    float reflection_intensity = 0.0f;
    float refraction_intensity = 1.0f;
    float refraction_index = 1.0f;

    Color pixel_color = Color(0.0f);
    bool reflects = false;
    bool refracts = false;

    // defined rays
    Ray reflection_ray;
    Ray refraction_ray;


}

#endif