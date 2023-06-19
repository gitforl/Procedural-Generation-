#include "FPFHDescriptorTester.hpp"

FPFHDescriptorTester::FPFHDescriptorTester(std::string objSrcPath):
    BaseDescriptor(objSrcPath) {
}

void FPFHDescriptorTester::CreateReferenceDescriptors(){

    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(gpuMesh, 1000000, 0);

    float supportRadius = 10.0f;

    referenceDescriptors = ShapeDescriptor::gpu::generateFPFHHistograms(
            pointCloud,
            gpuDescriptorOrigins,
            supportRadius);

    ShapeDescriptor::free::mesh(gpuMesh);
}

void FPFHDescriptorTester::CreateAlteredDescriptors(){

    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(alteredGpuMesh, 1000000, 0);

    float supportRadius = 10.0f;

    alteredDescriptors = ShapeDescriptor::gpu::generateFPFHHistograms(
        pointCloud,
        gpuDescriptorOrigins,
        supportRadius);

        ShapeDescriptor::free::mesh(gpuMesh);
}

void FPFHDescriptorTester::Compare(){
    auto comparisons = ShapeDescriptor::gpu::computeFPFHElementWiseEuclideanDistances(referenceDescriptors, alteredDescriptors);

    comparisonValues.clear();
    comparisonValues.reserve(comparisons.length);
    for(int i = 0; i < comparisons.length; i++)
        comparisonValues.push_back((float) comparisons.content[i]);

}