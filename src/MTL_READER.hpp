#ifndef MTL_READER_HPP
#define MTL_READER_HPP

#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include "Material.hpp"
#include "Texture.hpp"

class MTL_Reader {
private:




public:

    //CUDAVector<Material> material_vector;
    //CUDAVector<Texture> texture_vector;

    std::map <std::string, int> materialToId;
    std::map <std::string, int> textureToId;
    std::map <int, std::string> textureToId2;

    __host__ MTL_Reader(const std::string& filename, CUDAVector<Material>& material_vector, CUDAVector<Texture>& texture_vector)
    {
  
        //material_vector.reserve(100);
        readFile(filename, material_vector);
        readTextures(texture_vector);

    }   

    __host__ ~MTL_Reader()
    {}

    __host__ void readTextures(CUDAVector<Texture>& texture_vector) {

        texture_vector.reserve(textureToId.size());


        clock_t start_w, stop_w;
        start_w = clock();
        
        
        for (int i = 0; i < textureToId.size(); i++) {
            for (auto it = textureToId.begin(); it != textureToId.end(); ++it)
                if (it->second == i) {

                    std::cerr << "Loading Texture: " << it->first << " in id: " << it->second << ".\n";
                    texture_vector.emplace_back(it->first, it->second);
                }
                   
        }
        
        stop_w = clock();
        double timer_textures = ((double)(stop_w - start_w)) / CLOCKS_PER_SEC;

        std::cerr << "Loading Textures in  " << timer_textures << " seconds.\n";
        
        // vllt einfach an die pos pushen?
        /*
        std::map <std::string, int>::iterator it;
        for (it = textureToId.begin(); it != textureToId.end(); it++)
        {
            //texture_vector[it->second] = Texture(it->first, it->second);
            texture_vector.emplace_back(it->first, it->second);

            
            std::string filename = "Texture.ppm";

        }
        */
        
    }

    __host__ void readFile(const std::string& filename, CUDAVector<Material>& material_vector) {
        std::ifstream file_(filename);
        if (!file_.is_open()) {
            return;
        }
        std::string line_;

        Material tmp;
        std::string matName = "";

        while (getline(file_, line_)) {

            if (line_[0] == '#') continue;
            if (line_.empty()) continue;

            std::stringstream input(line_);
            std::string paramName;
            input >> paramName;

            if (paramName == "newmtl") {

                if (materialToId.size() > 0) {
                    material_vector.push_back(tmp);
                    std::cerr << "Material: " << tmp.id << " with texture id: " << tmp.textureid << ".\n";
                    tmp = Material();
                }

                input >> matName;
                std::cerr << "Reading: " << matName << std::endl;
                materialToId.insert(std::pair<std::string, int> {matName, material_vector.size()});
                tmp.id = material_vector.size();
                tmp.color = Color(1);

            }

            else if (paramName == "Ka") {

                float r, g, b;
                input >> r >> g >> b;
                tmp.ambient = Color(r, g, b);
            }

            else if (paramName == "Kd") {

                float r, g, b;
                input >> r >> g >> b;
                tmp.diffuse = Color(r, g, b);

            }

            else if (paramName == "Ks") {

                float r, g, b;
                input >> r >> g >> b;
                tmp.specular = Color(r, g, b);

            }

            else if (paramName == "Ns") {
                // shinyness
                float shinyness;
                input >> shinyness;
                tmp.shinyness = shinyness;

            }

            else if (paramName == "d") {
                // transparency
                float t;
                input >> t;
                tmp.transparent = (1 - t);
            }
            /*
            * Texture Maps
            */

            // TODO: set material attributes accordingly
            /*
            else if (paramName == "map_Kd") {
                std::string textureName;
                input >> textureName;
                if (textureName == "") continue;

                std::map<std::string, int>::iterator it;
                it = textureToId.find(textureName);
                if (it == textureToId.end()) {
                    textureToId[textureName] = textureToId.size();
                }

                tmp.mapD = true;
                tmp.textureid = textureToId[textureName];
            }
            */

            else if (paramName == "map_d") {
                std::string textureName;
                input >> textureName;
                if (textureName == "") continue;

                std::map<std::string, int>::iterator it;
                it = textureToId.find(textureName);
                if (it == textureToId.end()) {
                    textureToId[textureName] = textureToId.size();
                }
                tmp.mapTransparent = 1;
                tmp.textureid = textureToId[textureName];
            }

            else if (paramName == "map_kS") {
                std::string textureName;
                input >> textureName;
                if (textureName == "") continue;

                std::map<std::string, int>::iterator it;
                it = textureToId.find(textureName);
                if (it == textureToId.end()) {
                    textureToId[textureName] = textureToId.size();
                }
                tmp.mapS = 1;
                tmp.textureid = textureToId[textureName];
            }

            else if (paramName == "map_kA") {
                std::string textureName;
                input >> textureName;
                if (textureName == "") continue;

                std::map<std::string, int>::iterator it;
                it = textureToId.find(textureName);
                if (it == textureToId.end()) {
                    textureToId[textureName] = textureToId.size();
                }
                tmp.mapA = 1;
                tmp.textureid = textureToId[textureName];
            }

            else if (paramName == "map_Kd") {
                std::string textureName;
                input >> textureName;
                if (textureName == "") continue;

                std::map<std::string, int>::iterator it;
                it = textureToId.find(textureName);
                if (it == textureToId.end()) {
                    textureToId[textureName] = textureToId.size();
                }
                tmp.mapD = 1;
                tmp.textureid = textureToId[textureName];
            }

            // TODO:
            else if (paramName == "map_Ns") {
            }

            else continue;

        }

        material_vector.push_back(tmp);
        file_.close();
    }


    // export 
    __host__ std::map <std::string, int> getMaterialMap() {
        return materialToId;
    }
    __host__ std::map <std::string, int> getTextureMap() {
        return textureToId;
    }

};

#endif