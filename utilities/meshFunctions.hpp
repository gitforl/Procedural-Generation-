#pragma once

#include <utilities/aliases.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>

#include <unordered_map>

namespace MeshFunctions{
    struct IndexPair{
        size_t left;
        size_t right;
    };
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

    void ConstructMeshFromVisibleTriangles
    (
        ShapeDescriptor::cpu::Mesh &mesh,
        ShapeDescriptor::cpu::Mesh &outMesh,
        std::vector<bool> &triangleAppearsInImage,
        std::unordered_map<size_t, size_t> *mapping = nullptr
    );
}