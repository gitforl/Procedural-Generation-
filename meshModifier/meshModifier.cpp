#include "meshModifier.hpp"


void MeshModifier::ApplyNoiseToMesh(ShapeDescriptor::cpu::Mesh &mesh, float maxDistance, std::unordered_map<size_t, size_t> *mapping)
{
    auto vertexMap = MeshFunctions::MapVertexIndices(&mesh);
    MeshFunctions::MoveVerticesAlongAverageNormal(&mesh, vertexMap, maxDistance);
}


void ApplyOcclusionToMesh(ShapeDescriptor::cpu::Mesh &mesh, float maxDistance, std::unordered_map<size_t, size_t> *mapping)
{

}