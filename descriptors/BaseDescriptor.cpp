#include "BaseDescriptor.hpp"

BaseDescriptor::BaseDescriptor(std::string objSrcPath):
objSrcPath(objSrcPath) {
    InitializeMesh();
    InitializeAlteredMesh();
    InitializeDescriptorOrigins();

    vertexMap = MeshFunctions::MapVertexIndices(&mesh);
}

void BaseDescriptor::InitializeMesh() {
    mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
}

void BaseDescriptor::InitializeAlteredMesh() {
    alteredMesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);
}

void BaseDescriptor::InitializeDescriptorOrigins() {
    descriptorOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    gpuDescriptorOrigins = ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);
}

void BaseDescriptor::ApplyNoise(float noiseLevel){
    alteredMesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    MeshFunctions::MoveVerticesAlongAverageNormal(&alteredMesh, vertexMap, noiseLevel);
}

void BaseDescriptor::ComputeAverageDistance(){
    averageDistance = computeFloatAverage(comparisonValues);
}

void BaseDescriptor::ComputeStandardDeviation(){
    standardDeviation = computeFloatStandardDeviation(comparisonValues, averageDistance);
}

void BaseDescriptor::RunSingleNoiseTest(float noiseLevel){

    ApplyNoise(noiseLevel);
    CreateReferenceDescriptors();
    CreateAlteredDescriptors();
    Compare();
    ComputeAverageDistance();
    ComputeStandardDeviation();

    std::cout << "average distance: " << averageDistance << std::endl;
    std::cout << "standard deviation: " << standardDeviation << std::endl;
}

void BaseDescriptor::RunNoiseTestVaryingLevels(std::vector<float> noiseLevels){

    CreateReferenceDescriptors();

    for(float noiseLevel : noiseLevels){
        ApplyNoise(noiseLevel);
        CreateAlteredDescriptors();
        Compare();
        ComputeAverageDistance();
        ComputeStandardDeviation();

        std::cout << "average distance: " << averageDistance << std::endl;
        std::cout << "standard deviation: " << standardDeviation << std::endl;
    }
}