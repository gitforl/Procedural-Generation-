#pragma once

#include <utilities/aliases.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>

namespace MeshFunctions{
    struct boundingBox
    {
        ShapeDescriptor::cpu::float3 min, max;
        boundingBox(ShapeDescriptor::cpu::Mesh &mesh);
        ShapeDescriptor::cpu::float3 center();
        ShapeDescriptor::cpu::float3 span();
    };
    UIntVector FindSimilarVerticesIndices(unsigned int targetIndex, ShapeDescriptor::cpu::Mesh * mesh);
    void MoveVertexAlongNormal(ShapeDescriptor::cpu::Mesh * mesh);
    StringUIntMap MapVertexIndices(ShapeDescriptor::cpu::Mesh * mesh);
    StringFloat3Map VertexToAverageNormalMap(ShapeDescriptor::cpu::Mesh &mesh, StringUIntMap &indexMap);
    void MoveVerticesAlongAverageNormal(ShapeDescriptor::cpu::Mesh * mesh, StringUIntMap &indexMap, float maxDistance = 0.0f);
    void RecomputeVertices(ShapeDescriptor::cpu::Mesh &mesh, StringFloat3Map &normalMap);
}