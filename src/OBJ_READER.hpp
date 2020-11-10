#ifndef OBJ_READER_HPP
#define OBJ_READER_HPP

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

#include "CUDAVector.cuh"
#include "Vec4.hpp"
#include "Matrix.hpp"
#include "Texture.hpp"
#include "Material.hpp"

#include "Triangle.hpp"

#include "MTL_READER.hpp"




class OBJ_Reader {
private:

    bool hasMTLFile;

    int numVertices;
    int numVertexNormals;
    int numVertexTexture;
    
    int capacityV;
    
    Vec4* Vertices;
    Vec4* VertexNormals;
    Vec4* VertexTextures;

    int currentV;
    int currentVN;
    int currentT;


    DeviceVector<std::pair<int, int>> group_bounds;
    
    // for Material Mapping
    std::map <std::string, int> material2id;
    // for Texture Mapping
    std::map <std::string, int> texture2id;

    // resize Vertices Ptr
    __host__ void resizeV(const int& newsize) {

        Vec4* tmpVertices = new Vec4[newsize];
        Vec4* tmpVertexN = new Vec4[newsize];
        Vec4* tmpVertexT = new Vec4[newsize];

        for (int i = 0; i < currentV; ++i) {
            tmpVertices[i] = Vertices[i];
        }

        for (int i = 0; i < currentVN; ++i) {
            tmpVertexN[i] = VertexNormals[i];
        }

        for (int i = 0; i < currentT; ++i) {
            tmpVertexT[i] = VertexTextures[i];
        }


        Vertices = tmpVertices;
        VertexNormals = tmpVertexN;
        VertexTextures = tmpVertexT;
        capacityV = newsize;
    }

    __host__ void split(std::string str, const std::string& token, std::vector<std::string>& result) {
        while (str.size()) {
            int index = str.find(token);
            if (index != std::string::npos) {
                result.push_back(str.substr(0, index));
                str = str.substr(index + token.size());
                if (str.size() == 0)
                    result.push_back(str);
            }
            else {
                result.push_back(str);
                str = "";
            }
        }
    }

public:

    __host__ OBJ_Reader(const std::string& filename, CUDAVector<Triangle>& triangle_vector, CUDAVector<Material>& material_vector, CUDAVector<Texture>& texture_vector) {

        hasMTLFile = false;

        currentV = 0;
        currentVN = 0;
        currentT = 0;

        capacityV = 10000;

        // starting with 10000
        resizeV(capacityV);

        readFile(filename, triangle_vector, material_vector, texture_vector);

    }

    __host__ ~OBJ_Reader() {
        delete[] Vertices;
        delete[] VertexNormals;
        delete[] VertexTextures;
    }



    __host__ void readFile(const std::string& filename, CUDAVector<Triangle>& triangle_vector, CUDAVector<Material>& material_vector, CUDAVector<Texture>& texture_vector) {
        std::ifstream file_(filename);
        if (!file_.is_open()) {
            printf("OBJ file not found!");
            return;
        }
        //printf("Reading file!");
        std::string line_;

        Material curMtl;
        int curMtlId;
        int initialMtlId = material_vector.size();

        int triangleVectorSize = triangle_vector.size();
        int triangleVectorSizeEnd = triangle_vector.size();

        struct triangle_data {
            int vi, ni, ti;
            triangle_data(const int& vi, const int& ni, const int& ti) : 
                vi(vi),
                ni(ni),
                ti(ti) 
            {}
        };


        while (getline(file_, line_)) {

            if (line_[0] == '#') continue;
            if (line_.empty()) continue;


            // TODO: pass MAT
            if (line_[0] == 'f') {

                if (!hasMTLFile) curMtlId = 0;

                std::string tmp = line_.substr(2, line_.size() - 1);
                std::vector<std::string> vstrings;
                split(tmp, " ", vstrings);

                // building a vector of pairs
                std::vector<triangle_data> T;
                for (int i = 0; i < vstrings.size(); ++i) {
                    if (vstrings[i].size() != 0) {
                        std::vector<std::string> tmp2;
                        split(vstrings[i], "/", tmp2);
                        T.emplace_back(std::stoi(tmp2[0]), std::stoi(tmp2[2]), std::stoi(tmp2[1]));
                    }
                }

                // build n Triangles
                for (int i = 1; i < T.size() - 1; ++i) {

                    Vec4 p1 = Vertices[T[0].vi - 1];
                    Vec4 p2 = Vertices[T[i].vi - 1];
                    Vec4 p3 = Vertices[T[i + 1].vi - 1];

                    if (currentVN != 0) {
                        Vec4 n1 = VertexNormals[T[0].ni - 1];
                        Vec4 n2 = VertexNormals[T[i].ni - 1];
                        Vec4 n3 = VertexNormals[T[i + 1].ni - 1];

                        if (currentT != 0) {
                            Vec4 t1 = VertexTextures[T[0].ti - 1];
                            Vec4 t2 = VertexTextures[T[i].ti - 1];
                            Vec4 t3 = VertexTextures[T[i + 1].ti - 1];

                            if (curMtlId == 78) {
                                std::cerr << "";
                            }

                            triangle_vector.emplace_back(triangle_vector.size(), p1, p2, p3, n1, n2, n3, t1, t2, t3, curMtlId);
                        }
                        else {
                            triangle_vector.emplace_back(triangle_vector.size(), p1, p2, p3, n1, n2, n3, curMtlId);
                        }
                        
                    }
                    else {
                        triangle_vector.emplace_back(triangle_vector.size(), p1, p2, p3, curMtlId);
                        
                    }
                }

                continue;
            }

            std::stringstream input(line_);
            std::string paramName;
            input >> paramName;

            if (paramName == "v") {
                float x, y, z;
                input >> x >> y >> z;

                if (currentV == capacityV)
                    resizeV(capacityV * 2);

                Vertices[currentV] = Vec4(x, y, z, 1.0f);
                currentV++;
            }
            else if (paramName == "vn") {
                float x, y, z;
                input >> x >> y >> z;

                VertexNormals[currentVN] = Vec4(x, y, z, 0.0f);
                currentVN++;
            }
            else if (paramName == "g") {

                std::string object_name;
                input >> object_name;

                /*
                if (object_name != "wheel04_pivot") {
                    triangleVectorSize = triangle_vector.size();
                    continue;
                }
                */
                

                int currentEnd = triangle_vector.size();
                if (triangleVectorSize == currentEnd) continue;
                //std::pair<int, int> start_end = { triangleVectorSize , currentEnd };
                group_bounds.emplace_back(triangleVectorSize, currentEnd);
                triangleVectorSize = currentEnd;

            }
            else if (paramName == "vt") {
                float u, v;
                input >> u >> v;

                VertexTextures[currentT] = Vec4(u, v, 0.0f, 0.0f);
                currentT++;
            }

            else if (paramName == "mtllib") {


                // reading MTL File
                std::string mtl_file;
                input >> mtl_file;

                MTL_Reader mtl_reader(mtl_file, material_vector, texture_vector);

                material2id = mtl_reader.materialToId;
                texture2id = mtl_reader.textureToId;

                // init Materials
                hasMTLFile = true;
            }

            else if (paramName == "usemtl") {
                // set mat for incoming Triangles
                std::string mat_name;
                input >> mat_name;
                // get mat id from reader
                curMtlId = material2id[mat_name];
            }

            else continue;

        }

        int currentEnd = triangle_vector.size();
        if (triangleVectorSize != currentEnd) group_bounds.emplace_back(triangleVectorSize, currentEnd);

        file_.close();
    }



    __host__ int getNumGroups() {
        return group_bounds.size();
    }

    __host__ std::pair<int, int> getGroupBounds(const int& groupId) {
        return group_bounds[groupId];
    }
};


#endif