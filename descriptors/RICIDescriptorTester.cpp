#include "RICIDescriptorTester.hpp"

RICIDescriptorTester::RICIDescriptorTester(std::string objSrcPath):
    BaseDescriptor(objSrcPath) {
}

void RICIDescriptorTester::CreateReferenceDescriptors(){

    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    float supportRadius = 1.0f;

    referenceDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
        gpuMesh,
        gpuDescriptorOrigins,
        supportRadius);

    ShapeDescriptor::free::mesh(gpuMesh);
}

void RICIDescriptorTester::CreateAlteredDescriptors(){

    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);

    float supportRadius = 1.0f;

    alteredDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
        alteredGpuMesh,
        gpuDescriptorOrigins,
        supportRadius);

        ShapeDescriptor::free::mesh(gpuMesh);
}

void RICIDescriptorTester::Compare(){
    auto comparisons = ShapeDescriptor::gpu::computeRICIElementWiseModifiedSquareSumDistances(referenceDescriptors, alteredDescriptors);

    comparisonValues.clear();
    comparisonValues.reserve(comparisons.length);
    for(int i = 0; i < comparisons.length; i++)
        comparisonValues.push_back((float) comparisons.content[i]);

}