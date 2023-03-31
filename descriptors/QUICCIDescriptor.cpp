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

void QUICCIDescriptor::RankDescriptors(){

    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

    float supportRadius = 1.0f;

    auto testDescriptors = ShapeDescriptor::gpu::generateQUICCImages(
        gpuMesh,
        gpuDescriptorOrigins,
        supportRadius);

    ShapeDescriptor::free::mesh(gpuMesh);


    auto firstCopy = testDescriptors.content[0];
    for(int i = 1; i < testDescriptors.length; i++){
        auto content = testDescriptors.content[i];
        for(int j = 0; j < 32; j++)
            content.contents[j] = firstCopy.contents[j];
    }

    auto distances = ShapeDescriptor::gpu::computeQUICCIElementWiseWeightedHammingDistances(testDescriptors, alteredDescriptors);

    for(int i = 0; i < 10; i++){
        std::cout << distances.content[i] << std::endl;
    }
    
}