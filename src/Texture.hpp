#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include <string>
#include <fstream>

#include "CUDAVector.cuh"
#include "Color.hpp"
#include "tga.hpp"


class Texture {
private:

	int id;
	int width, height, channels;
	CUDAVector<CUDAVector<Color>> img;
	CUDAVector<Color> img2;

	__host__ void readTGA(const std::string& filename) {

		tga::TGA tga;
		if (!tga.Load(filename)) {
			std::cerr << "Error\n";
			return;
		}

		uint8_t* Data = tga.GetData();
		width = tga.GetWidth();
		height = tga.GetHeight();
		tga::ImageFormat Format = tga.GetFormat();

		channels = 0;
		if ((int)Format == 0) channels = 1;
		else if ((int)Format == 1) channels = 3;
		else if ((int)Format == 2) channels = 4;

		img.reserve(height);

		img2.reserve(height * width);

		size_t ab = sizeof(*Data);
		size_t cd = sizeof(uint8_t) * (height * width * channels);

		int dimxy = height * width;
		for (int w = 0; w < dimxy; w++) {

			int idx = w * channels;
			float r, g, b, a;
			r = float(Data[idx]) / 255.0f;
			g = float(Data[idx + 1]) / 255.0f;
			b = float(Data[idx + 2]) / 255.0f;
			if (channels == 4) a = (float)Data[idx + 3] / 255.0f;
			else a = 0.0f;
			img2.emplace_back(r, g, b, a);
		}



		for (int r = 0; r < height; ++r) {
			CUDAVector<Color> row;
			row.reserve(width);

			for (int c = 0; c < width; ++c) {

				int idx = r * height + c;

				float r, g, b, a;
				r = (float)Data[idx] / 255.0f;
				g = (float)Data[idx + 1] / 255.0f;
				b = (float)Data[idx + 2] / 255.0f;
				if (channels == 4) a = (float)Data[idx + 3] / 255.0f;
				else a = 0.0f;
				row.emplace_back(r, g, b, a);
				//img2.emplace_back(r, g, b, a);
			}
			img.push_back(row);
		}
	}

public:

	__host__ Texture() {
		channels = -1;
		height = -1;
		width = -1;
		id = -1;
	}

	__host__ Texture(const std::string& filename, const int& id) :
		id(id)
	{
		channels = -1;
		height = -1;
		width = -1;
		readTGA(filename);
		exportImg("Texturetest.ppm");

	}
	
	__host__ Texture(const Texture& t) :
		id(t.id),
		height(t.height),
		width(t.width),
		channels(t.channels)
	{
		img = t.img;
	}
	

	__host__ ~Texture() {
	}

	__host__ Texture operator =(const Texture& t) {
		id = t.id;
		width = t.width;
		height = t.height;
		channels = t.channels;
		img = t.img;
		return *this;
	}
	__host__ Texture operator =(const Texture&& t) {
		id = t.id;
		width = t.width;
		height = t.height;
		channels = t.channels;
		img = t.img;
		return *this;
	}

	__host__ __device__ Color mapColor(const float& u, const float& v) {



		// convert u & v into ints
		float u_copy = u;
		float v_copy = v;

		// check if out of bounds
		u_copy = u_copy > 1.0f ? 1.0f : u_copy;
		u_copy = u_copy < 0.0f ? 0.0f : u_copy;

		v_copy = v_copy > 1.0f ? 1.0f : v_copy;
		v_copy = v_copy < 0.0f ? 0.0f : v_copy;

		int w = u_copy * (width - 1);
		int h = v_copy * (height - 1);

		/*if (id == 7) {
			printf("Color w: %d h: %d\n", w, h);
		}
		*/

		//int idx = w * width + h;
		int idx = h * height + w;

		return img2[idx];
	}

	__host__ void exportImg(std::string&& filename) {

		std::ofstream out(filename);
		out << "P3\n" << width << ' ' << height << ' ' << "255\n";

		for (int w = width - 1; w >= 0; w--) {
			for (int h = 0; h < height; h++) {

				int idx = w * width + h;
				Color locColor(img2[idx]);
				locColor.clamp();
				out << int(255.99 * locColor.r) << ' '
					<< int(255.99 * locColor.g) << ' '
					<< int(255.99 * locColor.b) << '\n';
			}
		}		
	}

	__host__ __device__ float mapTransparency(const float& u, const float& v) {
		return 1.0f;
	}

};

#endif