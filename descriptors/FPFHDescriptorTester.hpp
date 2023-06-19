#pragma once

#include "BaseDescriptor.hpp"
#include <string>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramSearcher.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>

class FPFHDescriptorTester : public BaseDescriptor {
private:
    ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> referenceDescriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> alteredDescriptors;
public:
    FPFHDescriptorTester(std::string objSrcPath);
    void CreateReferenceDescriptors();
    void CreateAlteredDescriptors();
    void Compare();
};