#pragma once

#include <iostream>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>

#include <unordered_map>

namespace MeshModifier {
    
    void ApplyNoiseToMesh(ShapeDescriptor::cpu::Mesh &mesh, float maxDistance = 0.0f, std::unordered_map<size_t, size_t> *mapping = nullptr);
    void ApplyOcclusionToMesh(ShapeDescriptor::cpu::Mesh &mesh, float maxDistance = 0.0f, std::unordered_map<size_t, size_t> *mapping = nullptr);
    void ApplyClutterToMesh(ShapeDescriptor::cpu::Mesh &mesh, float maxDistance = 0.0f, std::unordered_map<size_t, size_t> *mapping = nullptr);

};