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
#include <shapeDescriptor/utilities/dump/descriptorImages.h>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>
#include <utilities/generalUtilities.hpp>

#include <utilities/descriptorDistance.cuh>

class QUICCIDescriptor : public BaseDescriptor {
private:
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> referenceDescriptors;
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> alteredDescriptors;
public:
    QUICCIDescriptor(std::string objSrcPath);
    QUICCIDescriptor(ShapeDescriptor::cpu::Mesh mesh);
    // void computeElementWiseDistances(BaseDescriptor &otherDescriptor);
    void FindElementWiseDistances(BaseDescriptor &otherDescriptor, ShapeDescriptor::gpu::array<IndexPair> &pairs);
    void FindMinDistances(BaseDescriptor &otherDescriptor);
    void FindDistances(BaseDescriptor &otherDescriptor);
    void CreateReferenceDescriptors();
    void CreateAlteredDescriptors();
    void CompareWithinDescriptors();
    void RankDescriptors();
    void Compare();
    void WriteDescriptorToImage(std::string imagePath);
};