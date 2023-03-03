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
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>

#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>

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

namespace QUICCI
{
    float ComputeAverageDistance(ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> comparisons)
    {
        float distanceSum = 0.0f;

        for (int i = 0; i < comparisons.length; i++)
        {
            distanceSum += comparisons.content[i].weightedHammingDistance;
        }

        return (distanceSum / comparisons.length);
    }

    float ComputeDistanceDeviation(ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> comparisons, float average)
    {
        float deviationSum = 0.0f;

        for (int i = 0; i < comparisons.length; i++)
        {
            float difference = (comparisons.content[i].weightedHammingDistance - average);
            deviationSum += (difference * difference);
        }

        return sqrt(deviationSum / comparisons.length);
    }

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> ComputeDeviceDescriptorsOfHostMesh(
        ShapeDescriptor::cpu::Mesh &mesh,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins,
        float supportRadius = 1.0f)
    {

        ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins =
            ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors =
            ShapeDescriptor::gpu::generateQUICCImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius);

        // Free memory
        ShapeDescriptor::free::array(gpuDescriptorOrigins);
        ShapeDescriptor::free::mesh(gpuMesh);

        return descriptors;
    }

    void runDistanceTestAtVaryingNoiseLevels(const std::string objSrcPath)
    {

        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
            ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

        auto result = QUICCI::ComputeDeviceDescriptorsOfHostMesh(mesh, spinOrigins);

        auto vertexMap = MeshFunctions::MapVertexIndices(&mesh);
        auto averageNormals = MeshFunctions::VertexToAverageNormalMap(mesh, vertexMap);

        unsigned int numDistances = 7;
        float maxNoiseDistances[] = {0.0f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f, 4.0f};

        for (unsigned int i = 0; i < numDistances; i++)
        {
            ShapeDescriptor::cpu::Mesh meshCopy = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
            MeshFunctions::MoveVerticesAlongAverageNormal(&meshCopy, vertexMap, maxNoiseDistances[i]);

            auto noisyResult = QUICCI::ComputeDeviceDescriptorsOfHostMesh(meshCopy, spinOrigins);

            auto comparisons = ShapeDescriptor::gpu::computeQUICCIElementWiseDistances(result, noisyResult);
            float averageDistance = QUICCI::ComputeAverageDistance(comparisons);
            std::cout << "average distance with noise (" << maxNoiseDistances[i] << "): " << averageDistance << std::endl;
            float distanceDeviation = QUICCI::ComputeDistanceDeviation(comparisons, averageDistance);
            std::cout << "standard deviation with noise (" << maxNoiseDistances[i] << "): " << distanceDeviation << std::endl;

            ShapeDescriptor::free::mesh(meshCopy);
            cudaFree(noisyResult.content);
        }
    }
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


void runSpinDescriptor(){
    const std::string objSrcPath = "../objects/T100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
        ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);


    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(gpuMesh, 1000000, 0);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins =
        ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);

    gpuDescriptorOrigins.length = 500;

    float supportRadius = 1.0f;

    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors =
            ShapeDescriptor::gpu::generateSpinImages(
                pointCloud,
                gpuDescriptorOrigins,
                supportRadius, 
                90.0f);

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors =
            ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    ShapeDescriptor::dump::descriptors(hostDescriptors, "../images/test2.png", true, 50);


}

void runRICIDescriptor(){
    const std::string objSrcPath = "../objects/T100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
        ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);


    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins =
        ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);

    gpuDescriptorOrigins.length = 500;

    float supportRadius = 1.0f;

    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
            ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius
                );

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors =
            ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    ShapeDescriptor::dump::descriptors(hostDescriptors, "../images/test3.png", true, 50);


}


int main()
{
    const std::string objSrcPath = "../objects/T100.obj";

    runRICIDescriptor();

    // QUICCI::runDistanceTestAtVaryingNoiseLevels(objSrcPath);
    // saveMeshCopyWithNoise(objSrcPath, 0.1f);
}