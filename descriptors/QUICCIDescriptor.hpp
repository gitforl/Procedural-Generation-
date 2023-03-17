#pragma once

#include "BaseDescriptor.hpp"
#include <string>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>

class QUICCIDescriptor : public BaseDescriptor {
private:
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> referenceDescriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> alteredDescriptors;
public:
    QUICCIDescriptor(std::string objSrcPath);
    void CreateReferenceDescriptors();
    void CreateAlteredDescriptors();
    void Compare();
};