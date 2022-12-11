#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <filesystem>
#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>

struct face {
    // vertices
    ShapeDescriptor::cpu::float3 v0;
    ShapeDescriptor::cpu::float3 v1;
    ShapeDescriptor::cpu::float3 v2;

    ShapeDescriptor::cpu::float3 normal;

    // neighbour faces
    face *n0 = nullptr; 
    face *n1 = nullptr;
    face *n2 = nullptr;

    face() = default;

    face(
        ShapeDescriptor::cpu::float3 v0,
        ShapeDescriptor::cpu::float3 v1,
        ShapeDescriptor::cpu::float3 v2,
        ShapeDescriptor::cpu::float3 normal
        ) : v0(v0), v1(v1), v2(v2), normal(normal) {}

};

ShapeDescriptor::cpu::float3 operator /= (ShapeDescriptor::cpu::float3 &target, float &other) {
    return target / other;
}

template <typename T>
void printVector(std::vector<T> vector, std::string headerText = ""){
    std::cout << headerText << std::endl;
    for(T element: vector){
        std::cout << element << std::endl;
    }
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> Compute(ShapeDescriptor::cpu::Mesh &mesh){
    // Load mesh
    // ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh("path/to/obj/file.obj", false);
        
    // Store it on the GPU
    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    // Define and upload descriptor origins
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins;
    descriptorOrigins.length = 1;
    descriptorOrigins.content = new ShapeDescriptor::OrientedPoint();//[1];
    descriptorOrigins.content->vertex = {77.9531, 11.5493, 2.79479};
    descriptorOrigins.content->normal = {0, 0, 1};

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins = 
        ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);

    // Compute the descriptor(s)
    float supportRadius = 1.0;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors = 
        ShapeDescriptor::gpu::generateQUICCImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius);
                
    // Copy descriptors to RAM
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors =
                ShapeDescriptor::copy::deviceArrayToHost(descriptors);
                    
    // Do something with descriptors here

    auto h = hostDescriptors.content->contents;

    // Free memory
    ShapeDescriptor::free::array(descriptorOrigins);
    // ShapeDescriptor::free::array(hostDescriptors);
    ShapeDescriptor::free::array(gpuDescriptorOrigins);
    ShapeDescriptor::free::array(descriptors);
    // ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::free::mesh(gpuMesh);

    return hostDescriptors;
}


int main()
{

    // const std::string objSrcPath = "../Dissimilarity-Tree-Reproduction/input/download/SHREC2016/SHREC2016_partial_retrieval/complete_objects/T100.obj";
    const std::string objSrcPath = "../objects/TN100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
 
    auto vertexMap = MeshFunctions::MapVertexIndices(&mesh);
    std::cout << "Map entries: " << vertexMap.size() << std::endl;

    auto averageNormals = MeshFunctions::VertexToAverageNormalMap(mesh, vertexMap);

    // MeshFunctions::RecomputeVertices(mesh, averageNormals);

    // auto mapElement = vertexMap[mesh.vertices[200].to_string()];
    // printVector(mapElement, "elements in " + mesh.vertices[200].to_string());

    // MoveVerticesAlongAverageNormal(&mesh, vertexMap);

    auto result = Compute(mesh);

    auto content = result.content->contents[0];

    const std::filesystem::path outpath = "../objects/outFine.obj";
    ShapeDescriptor::dump::mesh(mesh, outpath);
    ShapeDescriptor::free::mesh(mesh);

    std::cout << "Object vertexCount: " << mesh.vertexCount << std::endl;


}