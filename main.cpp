#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <filesystem>

#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>

#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>
#include <descriptors/BaseDescriptor.hpp>
#include <descriptors/QUICCIDescriptor.hpp>
#include <descriptors/RICIDescriptorTester.hpp>
#include <descriptors/SIDescriptorTester.hpp>
#include <descriptors/FPFHDescriptorTester.hpp>

#include <meshModifier/meshModifier.hpp>
#include <openglHandler/openglHandler.hpp>

#include <openglHandler/shader.hpp>


struct face
{
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
        ShapeDescriptor::cpu::float3 normal) : v0(v0), v1(v1), v2(v2), normal(normal) {}
};

ShapeDescriptor::cpu::float3 operator/=(ShapeDescriptor::cpu::float3 &target, float &other)
{
    return target / other;
}

template <typename T>
void printVector(std::vector<T> vector, std::string headerText = "")
{
    std::cout << headerText << std::endl;
    for (T element : vector)
    {
        std::cout << element << std::endl;
    }
}

std::vector<float> generateNoiseLevels(int numLevels, float multiplier = 2.0f, bool addZeroRef = true){
    
    std::vector<float> noiseLevels;

    if(addZeroRef)
        noiseLevels.push_back(0.0f);

    float currentDistance = 0.01f;

    for(int i = 0; i < numLevels; i++){
        currentDistance = currentDistance * multiplier;
        noiseLevels.push_back(currentDistance);
    }

    return noiseLevels;
}

void saveMeshCopyWithNoise(const std::string objSrcPath, float noiseMagnitude)
{

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    auto vertexMap = MeshFunctions::MapVertexIndices(&mesh);
    auto averageNormals = MeshFunctions::VertexToAverageNormalMap(mesh, vertexMap);

    MeshFunctions::MoveVerticesAlongAverageNormal(&mesh, vertexMap, noiseMagnitude);

    std::string objSuffix = ".obj";

    const std::filesystem::path outPath = objSrcPath.substr(0, objSrcPath.length() - objSuffix.length()) + "-NoiseRange-" + std::to_string(noiseMagnitude) + objSuffix;
    ShapeDescriptor::dump::mesh(mesh, outPath);
    ShapeDescriptor::free::mesh(mesh);
}


void run3DContextDescriptor()
{
    const std::string objSrcPath = "../objects/T100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
        ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(gpuMesh, 1000000, 0);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins =
        ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);

    gpuDescriptorOrigins.length = 500;

    float supportRadius = 10.0f;

    ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors =
        ShapeDescriptor::gpu::generate3DSCDescriptors(
            pointCloud,
            gpuDescriptorOrigins,
            0.2f,
            0.1f,
            2.5f);

    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> hostDescriptors =
        ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    // ShapeDescriptor::dump::descriptors(hostDescriptors, "../images/test3DSC.png", true, 50);
}

void FindMeshSpatialSpan(ShapeDescriptor::cpu::Mesh &mesh)
{
    ShapeDescriptor::cpu::float3 meshCenter = {0.0, 0.0, 0.0};

    for(unsigned int i = 0; i < mesh.vertexCount; i++)
    {
        meshCenter += mesh.vertices[i];   
    }

    meshCenter.x /= float(mesh.vertexCount);
    meshCenter.y /= float(mesh.vertexCount);
    meshCenter.z /= float(mesh.vertexCount);

    float maxDistance = 0.0f;
    float maxOriginDistance = 0.0f;
    ShapeDescriptor::cpu::float3 maxDistancePosition = {0.0, 0.0, 0.0};

    std::cout << "Center: " << meshCenter.x << ", " << meshCenter.y << ", " << meshCenter.z << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++)
    {
        float originDistance = length(mesh.vertices[i]);
        if(originDistance > maxOriginDistance)
        {
            maxOriginDistance = originDistance;
        }
        auto vertexDistanceFromMeshCenter = mesh.vertices[i] - meshCenter;
        float distance = length(vertexDistanceFromMeshCenter);
        if( distance > maxDistance)
        {
            maxDistance = distance;
            maxDistancePosition = mesh.vertices[i];
        }  
    }

    std::cout << "max: " << maxDistance << std::endl;
    std::cout << "origin max: " << maxOriginDistance << std::endl;
    // std::cout << "Center: " << meshCenter.x << ", " << meshCenter.y << ", " << meshCenter.z << std::endl;

}

int main()
{
    const std::string objSrcPath = "../objects/T100.obj";
    const std::string t34SrcPath = "../objects/T34.obj";
    // const std::string objSrcPath = "../objects/OCCLUDED.obj";
    
    // QUICCIDescriptor testDescriptor(objSrcPath);

    // // testDescriptor.MeshSelfIntersects();
    // testDescriptor.CGALMeshTest();

    // MeshModifier meshModifier(objSrcPath);
    // meshModifier.DrawScreen();

    std::cout << "her" << std::endl;

    auto mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    auto t34 = ShapeDescriptor::utilities::loadOBJ(t34SrcPath, true);

    OpenGLHandler openGLHandler;
    openGLHandler.AddMesh(mesh);
    // openGLHandler.AddMesh(t34);
    openGLHandler.Draw();
    // openGLHandler.CreateMeshFromVisibleTriangles();

    // FindMeshSpatialSpan(mesh);

    // meshModifier.CheckMesh();

    // std::vector<float> noiseLevels({0.0f, 0.01f, 0.1f, 1.0f});

    // testDescriptor.RunNoiseTestAtLevel(0.0f);
    // testDescriptor.RankDescriptors();
    // testDescriptor.RunNoiseTestAtLevel(0.1f);

    // testDescriptor.RankDescriptors();


    // testDescriptor.RunNoiseTestAtVaryingLevels(noiseLevels);

    std::cout << "closing program" << std::endl;

}