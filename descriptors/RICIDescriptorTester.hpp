#pragma once

#include "BaseDescriptor.hpp"
#include <string>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>

class RICIDescriptorTester : public BaseDescriptor {
private:
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> referenceDescriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> alteredDescriptors;
public:
    RICIDescriptorTester(std::string objSrcPath);
    void CreateReferenceDescriptors();
    void CreateAlteredDescriptors();
    void Compare();
};