#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <shapeDescriptor/cpu/types/float3.h>

using UIntVector = std::vector<unsigned int>;
using StringUIntMap = std::map<std::string, UIntVector>;
using StringFloat3Map = std::map<std::string, ShapeDescriptor::cpu::float3>;