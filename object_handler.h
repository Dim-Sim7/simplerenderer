// loadobj.h
#pragma once
#include <string>
#include <vector>
#include "geometry.h"


bool loadObj(const std::string& filename,
             std::vector<vec3>& vertices,
             std::vector<Face>& faces);
