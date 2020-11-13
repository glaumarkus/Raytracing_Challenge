// std includes
#include <iostream>
#include <time.h>
#include <string>
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

#include "World.cuh"

#include "Sphere.hpp"
#include "Plane.hpp"
#include "Cube.hpp"
#include "Triangle.hpp"

#include "OBJ_READER.hpp"

#include "CUDAVector.cuh"

#include "Texture.hpp"


#define DEBUG false
#define RENDER true
#define PRINTIMG false
#define SAVING true
#define RUN true


#define MAX_DEPTH 3
__global__ void renderRefraction(CUDAVector<Image_buffer>& image_buffer, int max_x, int max_y, World& world) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    // only render if it has been set
    if (image_buffer[pixel_index].renderRefraction()) {
        Ray ray(image_buffer[pixel_index].refraction.origin, image_buffer[pixel_index].refraction.direction);
        Intersection intersection;
        world.intersect(intersection, ray);

        if (intersection.hit()) {
            Comps comps(world.prepare_computations(intersection, ray));
            world.color_at3(comps2, image_buffer[pixel_index]);
        } 

        else {
            image_buffer[pixel_index].setRefractFalse();
        }
    }
}

// maybe also works with device vectors ? 
__global__ void renderWorld2(CUDAVector<Image_buffer>& image_buffer, int max_x, int max_y, World& world) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    // Camera Ray
    float X = 2.0f * float(i) / float(max_x) - 1;
    float Y = -2.0f * float(j) / float(max_y) + 1;

    Ray ray(world.camera->getRay(X, Y));

    // checking for Intersections
    Intersection intersection;
    world.intersect(intersection, ray);

    if (intersection.hit()) {
        Comps comps2(world.prepare_computations(intersection, ray));
        world.color_at3(comps2, image_buffer[pixel_index]);
    }
}




void CUDAmain_run2(const std::string& filename) {

    World* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(World*)));
    world->createWorld(filename);

    while (true) {

        // threads & blocks
        int tx = 8;
        int ty = 8;

        // image params from World
        int nx = world->getWidth();
        int ny = world->getHeight();

        int num_pixels = nx * ny;
        CUDAVector<Image_buffer> image_buffer;
        image_buffer.reserve(num_pixels);

        // render image
        if (RENDER) {

            // render initial state
            renderWorld2 << <blocks, threads >> > (image_buffer, nx, ny, *world);
            
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // now render refractions
            for (int numRefractions = 0; numRefractions < MAX_DEPTH; numRefractions++) {
                renderRefraction << <blocks, threads >> > (image_buffer, nx, ny, *world);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }
            
        }

        // save image
        if (SAVING) {
            std::ofstream out("out.ppm");
            out << "P3\n" << nx << " " << ny << "\n255\n";
            for (int j = ny - 1; j >= 0; j--) {
                for (int i = 0; i < nx; i++) {
                    int pixel_index = j * nx + i;
                    int ir = int(255.99 * image_buffer[pixel_index].color.r);
                    int ig = int(255.99 * image_buffer[pixel_index].color.g);
                    int ib = int(255.99 * image_buffer[pixel_index].color.b);
                    out << ir << " " << ig << " " << ib << "\n";
                }
            }
        }

        // what next?
        char input[10];

        std::cerr << "What action now?\n exit: programm stops\ncamera: load new camera\nworld: reload world\n";
        std::cin >> input;

        if (strcmp(input, "camera") == 0) {

            std::ifstream file_("camera.txt");
            if (!file_.is_open()) {
                printf("Camera file not found!");
                break;
            }
            std::string line_;
            while (getline(file_, line_)) {
                if (line_[0] == '#') continue;
                if (line_.empty()) continue;
                std::stringstream input(line_);
                std::string paramName;
                input >> paramName;
                if (paramName == "Camera:") {
                    float px, py, pz, fx, fy, fz, ux, uy, uz, fov;
                    input >> px >> py >> pz >> fx >> fy >> fz >> ux >> uy >> uz >> fov;

                    Vec4 origin(px, py, pz, 1.0f);
                    Vec4 forward(fx, fy, fz);
                    Vec4 upguide(ux, uy, uz);
                    fov = fov * PI / 180.f;
                    world->camera->createCamera(origin, forward, upguide, fov, aspectRatio);
                    std::cerr << "Adjusted Camera\n";
                    break;
                }
            
            }
        }

    }


}

/*
RENDER FUNCTION
*/
__global__ void renderWorld(Color* fb_color, int max_x, int max_y, World& world) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    // Camera Ray
    float X = 2.0f * float(i) / float(max_x) - 1;
    float Y = -2.0f * float(j) / float(max_y) + 1;
    Ray ray(world.camera->getRay(X, Y));

    //if (pixel_index != 0) return;


    // checking for Intersections
    Intersection intersection;
    world.intersect(intersection, ray);

    if (intersection.hit()) {
        
        Comps comps2;
        comps2 = world.prepare_computations(intersection, ray);

        // Color to change
        Color color(0);

        world.color_at2(comps2, color);
        fb_color[pixel_index] = color;
    }
    else {
        fb_color[pixel_index] = Color(0);
    }
}


/*

MAIN FUNCTION -> 1st argument is supposed to be Input Scene

*/

void CUDAmain(const std::string& filename) {


    clock_t start_w, stop_w;
    start_w = clock();

    World* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(World*)));
    world->createWorld(filename);

    stop_w = clock();
    double timer_world = ((double)(stop_w - start_w)) / CLOCKS_PER_SEC;

    std::cerr << "Loading World in  " << timer_world << " seconds.\n";


    world->print();

    Triangle t1 = world->triangle_vector[0];
    Triangle t2 = world->triangle_vector[1];

    // image params from World
    int nx = world->getWidth();
    int ny = world->getHeight();

    // threads & blocks
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_color_size = num_pixels * sizeof(Color);

    // allocate FB with color
    Color* fb_color;
    checkCudaErrors(cudaMallocManaged((void**)&fb_color, fb_color_size));

    // utility
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    // render image
    if (RENDER) renderWorld << <blocks, threads >> > (fb_color, nx, ny, *world);
    //if (RENDER) renderWorld << <blocks, threads >> > (fb_color, nx, ny, world);

    // sync
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    //__host__ void exportImg(std::string && filename)
    if (SAVING) {
        std::ofstream out("out.ppm");
        out << "P3\n" << nx << " " << ny << "\n255\n";

        for (int j = ny - 1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * fb_color[pixel_index].r);
                int ig = int(255.99 * fb_color[pixel_index].g);
                int ib = int(255.99 * fb_color[pixel_index].b);
                out << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // Output FB_Color as Image, change later to report std::cout   
    if (PRINTIMG) {
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny - 1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * fb_color[pixel_index].r);
                int ig = int(255.99 * fb_color[pixel_index].g);
                int ib = int(255.99 * fb_color[pixel_index].b);
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // Free allocated Memory
    checkCudaErrors(cudaFree(fb_color));
    checkCudaErrors(cudaFree(world));
}


void CUDAmain_run(const std::string& filename) {

    // init world
    clock_t start_w, stop_w;
    start_w = clock();

    World* world;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(World*)));
    world->createWorld(filename);

    stop_w = clock();
    double timer_world = ((double)(stop_w - start_w)) / CLOCKS_PER_SEC;

    std::cerr << "Loading World in  " << timer_world << " seconds.\n";
    world->print();

    // allocate FB with color
    Color* fb_color;
    


    while (true) {

        // image params from World
        int nx = world->getWidth();
        int ny = world->getHeight();

        // threads & blocks
        int tx = 8;
        int ty = 8;

        std::cerr << "Rendering a " << nx << "x" << ny << " image ";
        std::cerr << "in " << tx << "x" << ty << " blocks.\n";

        int num_pixels = nx * ny;
        size_t fb_color_size = num_pixels * sizeof(Color);

        checkCudaErrors(cudaMallocManaged((void**)&fb_color, fb_color_size));

        clock_t start, stop;
        start = clock();

        // Render our buffer
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);

        // render image
        if (RENDER) renderWorld << <blocks, threads >> > (fb_color, nx, ny, *world);

        // sync
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (SAVING) {

            std::ofstream out("out.ppm");
            out << "P3\n" << nx << " " << ny << "\n255\n";

            for (int j = ny - 1; j >= 0; j--) {
                for (int i = 0; i < nx; i++) {
                    size_t pixel_index = j * nx + i;
                    int ir = int(255.99 * fb_color[pixel_index].r);
                    int ig = int(255.99 * fb_color[pixel_index].g);
                    int ib = int(255.99 * fb_color[pixel_index].b);
                    out << ir << " " << ig << " " << ib << "\n";
                }
            }
        }

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";

        checkCudaErrors(cudaFree(fb_color));

        // what next?
        char input[10];

        std::cerr << "What action now?\n exit: programm stops\ncamera: load new camera\nworld: reload world\n";
        std::cin >> input;

        if (strcmp(input, "exit") == 0) {
            break;
        }

        if (strcmp(input, "camera") == 0) {

            std::ifstream file_("camera.txt");
            if (!file_.is_open()) {
                printf("Camera file not found!");
                break;
            }
            std::string line_;
            while (getline(file_, line_)) {
                if (line_[0] == '#') continue;
                if (line_.empty()) continue;
                std::stringstream input(line_);
                std::string paramName;
                input >> paramName;
                if (paramName == "Camera:") {
                    float px, py, pz, fx, fy, fz, ux, uy, uz, fov;
                    input >> px >> py >> pz >> fx >> fy >> fz >> ux >> uy >> uz >> fov;

                    Vec4 origin(px, py, pz, 1.0f);
                    Vec4 forward(fx, fy, fz);
                    Vec4 upguide(ux, uy, uz);
                    fov = fov * PI / 180.f;
                    world->camera->createCamera(origin, forward, upguide, fov, aspectRatio);
                    std::cerr << "Adjusted Camera\n";
                    break;
                }
            
            }
        }

    }
    // Free allocated Memory
    checkCudaErrors(cudaFree(fb_color));
    checkCudaErrors(cudaFree(world));

}

int main(int argc, char* argv[]) {

    if (DEBUG) {

        std::string texturepath = "textures/sls_interior.tga";
        Texture t(texturepath,0);

        std::string filename = "textures/copy.tga";
        t.exportImg("textures/copy.tga");
        return 0;
    }

    if (RUN) {
        std::string filename = "Scene.txt";
        CUDAmain_run(filename);
        return 0;
    }

    

    if (false) {
        if (argc == 1) {
            printf("No Input File was specified!\n");
            return 0;
        }
        std::string filename = argv[1];
        CUDAmain(filename);
    }

    else {

        std::string filename = "Scene.txt";
        CUDAmain(filename);
    }

    return 0;
}
