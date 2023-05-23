#pragma once

#include <bitset>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>


const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;

namespace DescriptorDistance {

    namespace Utilities {
        inline unsigned int GetQUICCIChunk(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex);
        inline unsigned int CountBitsTrueInDescriptors(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex);
    }

    namespace Hamming {

        struct Weights {
            float setBitPenalty;
            float unsetBitPenalty;
        };

        inline Weights ComputeWeights(unsigned int setBitCount, unsigned int totalBitsInBitString);

        float Compute(
            ShapeDescriptor::QUICCIDescriptor* needleDescriptors, 
            ShapeDescriptor::QUICCIDescriptor* haystackDescriptors,
            const unsigned int numNeedleDescriptors,
            const unsigned int numHaystackDescriptors
            );
        float ComputeElementWiseDistance(
            ShapeDescriptor::QUICCIDescriptor* descriptors,
            const unsigned int numDescriptors
            );
        float ComputeAgainstSelfOnChunk(
            ShapeDescriptor::QUICCIDescriptor* descriptors,
            const unsigned int descriptorIndex,
            const unsigned int chunkIndex
            );
        inline float computeIndividual(const Weights hammingWeights, const unsigned int needle,  const unsigned int haystack);
        inline unsigned int CountBitsTrueLeftFalseRight(const unsigned int left, const unsigned int right);
        inline unsigned int CountBitsFalseLeftTrueRight(const unsigned int left, const unsigned int right);
    }
}