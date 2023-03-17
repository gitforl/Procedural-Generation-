#include "QUICCIDescriptor.hpp"

QUICCIDescriptor::QUICCIDescriptor(std::string objSrcPath):
    BaseDescriptor(objSrcPath) {
}

void QUICCIDescriptor::CreateReferenceDescriptors(){

    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    float supportRadius = 1.0f;

    referenceDescriptors = ShapeDescriptor::gpu::generateQUICCImages(
        gpuMesh,
        gpuDescriptorOrigins,
        supportRadius);

    ShapeDescriptor::free::mesh(gpuMesh);
}

void QUICCIDescriptor::CreateAlteredDescriptors(){

    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);

    float supportRadius = 1.0f;

    alteredDescriptors = ShapeDescriptor::gpu::generateQUICCImages(
        alteredGpuMesh,
        gpuDescriptorOrigins,
        supportRadius);

        ShapeDescriptor::free::mesh(gpuMesh);
}

void QUICCIDescriptor::Compare(){
    auto comparisons = ShapeDescriptor::gpu::computeQUICCIElementWiseDistances(referenceDescriptors, alteredDescriptors);

    comparisonValues.clear();
    comparisonValues.reserve(comparisons.length);
    for(int i = 0; i < comparisons.length; i++)
        comparisonValues.push_back((float) comparisons.content[i].weightedHammingDistance);

}