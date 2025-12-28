#include "object_handler.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool loadObj(const std::string& filename, std::vector<vec3>& vertices, std::vector<Face>& faces) {

    std::ifstream file(filename);


    if(!file.is_open()){
        std::cerr << "Error: cannot open OBJ file: " << filename << "\n";
        return false; //handle error
    }

    std::string token;
    std::string line;
    while (std::getline(file, line)) {

        // -------------------------
        // Parse vec3
        // -------------------------
        if (line.rfind("v ", 0) == 0) {
            std::stringstream ss(line.substr(2));
            vec3 v;
            if (!(ss >> v.x >> v.y >> v.z)) {
                std::cerr << "Warning: malformed vec3 line\n";
                continue;
            }
            vertices.push_back(v);
        }
        // -------------------------
        // Parse triangular face
        // -------------------------
        else if (line.rfind("f ", 0) == 0) { //check if line starts with "v "
            Face f;
            std::stringstream ss(line.substr(2)); // e.g. 206/199/206 242/244/242 208/180/208
            for (int i = 0; i < 3; i++) {
                if (!(ss >> token)) {
                    std::cerr << "Warning: non-triangle face or malformed\n";
                    break;
                }
            int v = 0, vt = 0, vn = 0;

            size_t p1 = token.find('/'); //find('/') returns the position of the first '/'.
            size_t p2 = token.rfind('/'); // rfind('/') finds the last '/' in the string.

            if (p1 == std::string::npos) { //no slash
                //format v
                v = std::stoi(token);
            }
            else if (p1 == p2) { //one slash
                v  = std::stoi(token.substr(0, p1));
                vt = std::stoi(token.substr(p1 + 1));
            }
            else { //two slash
                v = std::stoi(token.substr(0, p1));
                if (p2 > p1 + 1)
                    vt = std::stoi(token.substr(p1 + 1, p2 - p1 - 1));
                vn = std::stoi(token.substr(p2 + 1));
            }

                f.v[i]  = v - 1;
                f.vt[i] = vt - 1;
                f.vn[i] = vn - 1;
            }
            faces.push_back(f);

        }
    }
    file.close();
    return true;

}