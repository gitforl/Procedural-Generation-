#include "SIDescriptorTester.hpp"

SIDescriptorTester::SIDescriptorTester(std::string objSrcPath):
    BaseDescriptor(objSrcPath) {
}

void SIDescriptorTester::CreateReferenceDescriptors(){

    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(gpuMesh, 1000000, 0);

    float supportRadius = 10.0f;

    referenceDescriptors = ShapeDescriptor::gpu::generateSpinImages(
            pointCloud,
            gpuDescriptorOrigins,
            supportRadius,
            90.0f);

    ShapeDescriptor::free::mesh(gpuMesh);
}

void SIDescriptorTester::CreateAlteredDescriptors(){

    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(alteredGpuMesh, 1000000, 0);

    float supportRadius = 10.0f;

    alteredDescriptors = ShapeDescriptor::gpu::generateSpinImages(
        pointCloud,
        gpuDescriptorOrigins,
        supportRadius,
        90.0f);

        ShapeDescriptor::free::mesh(gpuMesh);
}

void SIDescriptorTester::Compare(){
    auto comparisons = ShapeDescriptor::gpu::computeSIElementWiseEuclideanDistances(referenceDescriptors, alteredDescriptors);

    comparisonValues.clear();
    comparisonValues.reserve(comparisons.length);
    for(int i = 0; i < comparisons.length; i++)
        comparisonValues.push_back((float) comparisons.content[i]);

}