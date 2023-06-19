#pragma once

#include "BaseDescriptor.hpp"
#include <string>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>

class SIDescriptorTester : public BaseDescriptor {
private:
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> referenceDescriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> alteredDescriptors;
public:
    SIDescriptorTester(std::string objSrcPath);
    void CreateReferenceDescriptors();
    void CreateAlteredDescriptors();
    void Compare();
};