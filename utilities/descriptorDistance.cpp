#include "descriptorDistance.hpp"

inline unsigned int DescriptorDistance::Utilities::GetQUICCIChunk(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex)
{
    return image[imageIndex].contents[chunkIndex];
}

inline unsigned int DescriptorDistance::Utilities::CountBitsTrueInDescriptors(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex)
{
    unsigned int count = 0;

    for (int chunkIndex = 0; chunkIndex < uintsPerQUICCImage; chunkIndex += 32)
    {
        unsigned int chunk = Utilities::GetQUICCIChunk(image, imageIndex, chunkIndex);
        unsigned int chunkBitCount = std::bitset<32>(chunk).count();

        #pragma omp atomic
        count += chunkBitCount;
    }

    return count;
}

inline DescriptorDistance::Hamming::Weights DescriptorDistance::Hamming::ComputeWeights(unsigned int setBitCount, unsigned int totalBitsInBitString)
{
    unsigned int queryImageUnsetBitCount = totalBitsInBitString - setBitCount;

    // If any count is 0, bump it up to 1

    #ifdef __CUDACC__
        setBitCount = max(setBitCount, 1);
        queryImageUnsetBitCount = max(queryImageUnsetBitCount, 1);
    #else
        setBitCount = std::max<unsigned int>(setBitCount, 1);
        queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);
    #endif

    // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
    float missedSetBitPenalty = float(totalBitsInBitString) / float(setBitCount);
    float missedUnsetBitPenalty = float(totalBitsInBitString) / float(queryImageUnsetBitCount);

    return {missedSetBitPenalty, missedUnsetBitPenalty};

}

float DescriptorDistance::Hamming::Compute(
    ShapeDescriptor::QUICCIDescriptor* needleDescriptors, 
    ShapeDescriptor::QUICCIDescriptor* haystackDescriptors,
    const unsigned int numNeedleDescriptors,
    const unsigned int numHaystackDescriptors
    )
{

    float distance = 0.0f;

    #pragma omp parallel for
    for(unsigned int needleIndex = 0; needleIndex < numNeedleDescriptors; needleIndex++)
    {
        unsigned int referenceImageBitCount = Utilities::CountBitsTrueInDescriptors(&needleDescriptors[needleIndex], 0);
        Weights weights =  ComputeWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

        for(unsigned int haystackIndex = 0; haystackIndex < numNeedleDescriptors; haystackIndex++)
        {
            for (int chunk = 0; chunk < uintsPerQUICCImage; chunk += 32)
            {
                unsigned int needleChunk = Utilities::GetQUICCIChunk(needleDescriptors, needleIndex, chunk);
                unsigned int haystackChunk = Utilities::GetQUICCIChunk(haystackDescriptors, haystackIndex, chunk);

                float distanceBetweenChunks = computeIndividual(weights, needleChunk, haystackChunk);                
                #pragma omp atomic
                distance += distanceBetweenChunks;
            }
        }
    }
    
    return distance;
}

float DescriptorDistance::Hamming::ComputeElementWiseDistance(
    ShapeDescriptor::QUICCIDescriptor* descriptors,
    const unsigned int numDescriptors
    )
{
    unsigned int referenceImageBitCount = Utilities::CountBitsTrueInDescriptors(&descriptors[0], 0);
    Weights weights =  ComputeWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

    float distance = 0.0f;

    #pragma omp parallel for

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        for (int chunk = 0; chunk < uintsPerQUICCImage; chunk += 32)
        {
            unsigned int needleChunk = Utilities::GetQUICCIChunk(descriptors, i, chunk);
            unsigned int haystackChunk = Utilities::GetQUICCIChunk(descriptors, i, chunk);

            float distanceBetweenChunks = computeIndividual(weights, needleChunk, haystackChunk);

            #pragma omp atomic
            distance += distanceBetweenChunks;
        }
    }

    return distance;
}

float DescriptorDistance::Hamming::ComputeAgainstSelfOnChunk(
    ShapeDescriptor::QUICCIDescriptor* descriptors,
    const unsigned int descriptorIndex,
    const unsigned int chunkIndex
    )
{
    unsigned int chunk = Utilities::GetQUICCIChunk(descriptors, descriptorIndex, chunkIndex);

    float distance = computeIndividual({1.0f, 1.0f}, chunk, chunk);

    return distance;

}

inline float DescriptorDistance::Hamming::computeIndividual(const Weights hammingWeights, const unsigned int needle,  const unsigned int haystack)
{
    const unsigned int numberOfBitsTrueInNeedleFalseInHaystack = CountBitsTrueLeftFalseRight(needle, haystack);
    const unsigned int numberOfBitsFalseInNeedleTrueInHaystack = CountBitsFalseLeftTrueRight(needle, haystack);

    const float distance =  float(numberOfBitsTrueInNeedleFalseInHaystack) * hammingWeights.setBitPenalty
                         +  float(numberOfBitsFalseInNeedleTrueInHaystack) * hammingWeights.unsetBitPenalty;

    return distance;
}

inline unsigned int DescriptorDistance::Hamming::CountBitsTrueLeftFalseRight(const unsigned int left, const unsigned int right)
{
    return std::bitset<32>(left & !right).count();
}

inline unsigned int DescriptorDistance::Hamming::CountBitsFalseLeftTrueRight(const unsigned int left, const unsigned int right)
{
    return std::bitset<32>(!left & right).count();
}
