#include "../utilities/descriptorDistance.cuh"
#include "utility.cuh"

namespace RICIUTILITES
{

    const int indexBasedWarpCount = 16;

    __device__ int computeImageSquaredSumGPU(const ShapeDescriptor::RICIDescriptor &needleImage)
    {

        const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
        const int laneIndex = threadIdx.x % 32;

        unsigned int threadSquaredSum = 0;

        static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");

        // Scores are computed one row at a time.
        // We differentiate between rows to ensure the final pixel of the previous row does not
        // affect the first pixel of the next one.
        for (int pixel = 0; pixel < spinImageElementCount; pixel++)
        {
            radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                needleImage.contents[pixel];

            int targetThread;
            if (laneIndex > 0)
            {
                targetThread = laneIndex - 1;
            }
            else if (pixel > 0)
            {
                targetThread = 31;
            }
            else
            {
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;

            if (laneIndex == 31)
            {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
            }
            else
            {
                threadNeedleValue = currentNeedlePixelValue;
            }

            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);
            int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

            threadSquaredSum += unsigned(needleDelta * needleDelta);
        }

        int squaredSum = warpAllReduceSum(threadSquaredSum);

        return squaredSum;
    }

    __device__ size_t compareConstantRadialIntersectionCountImagePairGPU(
        const ShapeDescriptor::RICIDescriptor *needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::RICIDescriptor *haystackImages,
        const size_t haystackImageIndex)
    {

        const int laneIndex = threadIdx.x % 32;

        // Assumption: there will never be an intersection count over 65535 (which would cause this to overflow)
        size_t threadDeltaSquaredSum = 0;

        static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");

        // Scores are computed one row at a time.
        // We differentiate between rows to ensure the final pixel of the previous row does not
        // affect the first pixel of the next one.
        for (int row = 0; row < spinImageWidthPixels; row++)
        {
            // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
            for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize)
            {
                radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    needleImages[needleImageIndex].contents[row * spinImageWidthPixels + pixel];
                radialIntersectionCountImagePixelType currentHaystackPixelValue =
                    haystackImages[haystackImageIndex].contents[row * spinImageWidthPixels + pixel];

                // This bit handles the case where an image is completely constant.
                // In that case, we use the absolute sum of squares as a distance function instead
                int imageDelta = int(currentNeedlePixelValue) - int(currentHaystackPixelValue);
                threadDeltaSquaredSum += unsigned(imageDelta * imageDelta); // TODO: size_t?
            }
        }

        // image is constant.
        // In those situations, imageScore would always be 0
        // So we use an unfiltered squared sum instead
        size_t imageScore = warpAllReduceSum(threadDeltaSquaredSum);

        return imageScore;
    }

    __device__ int compareRadialIntersectionCountImagePairGPU(
        const ShapeDescriptor::RICIDescriptor *needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::RICIDescriptor *haystackImages,
        const size_t haystackImageIndex,
        const int distanceToBeat = INT_MAX)
    {

        int threadScore = 0;
        const int laneIndex = threadIdx.x % 32;

        static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");

        // Scores are computed one row at a time.
        // We differentiate between rows to ensure the final pixel of the previous row does not
        // affect the first pixel of the next one.
        for (int row = 0; row < spinImageWidthPixels; row++)
        {
            radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;
            radialIntersectionCountImagePixelType previousWarpLastHaystackPixelValue = 0;
            // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
            for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize)
            {
                radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    needleImages[needleImageIndex].contents[row * spinImageWidthPixels + pixel];
                radialIntersectionCountImagePixelType currentHaystackPixelValue =
                    haystackImages[haystackImageIndex].contents[row * spinImageWidthPixels + pixel];

                // To save on memory bandwidth, we use shuffle instructions to pass around other values needed by the
                // distance computation. We first need to use some logic to determine which thread should read from which
                // other thread.
                int targetThread;
                if (laneIndex > 0)
                {
                    // Each thread reads from the previous one
                    targetThread = laneIndex - 1;
                }
                // For these last two: lane index is 0. The first pixel of each row receives special treatment, as
                // there is no pixel left of it you can compute a difference over
                else if (pixel > 0)
                {
                    // If pixel is not the first pixel in the row, we read the difference value from the previous iteration
                    targetThread = 31;
                }
                else
                {
                    // If the pixel is the leftmost pixel in the row, we give targetThread a dummy value that will always
                    // result in a pixel delta of zero: our own thread with ID 0.
                    targetThread = 0;
                }

                radialIntersectionCountImagePixelType threadNeedleValue = 0;
                radialIntersectionCountImagePixelType threadHaystackValue = 0;

                // Here we determine the outgoing value of the shuffle.
                // If we're the last thread in the warp, thread 0 will read from us if we're not processing the first batch
                // of 32 pixels in the row. Since in that case thread 0 will read from itself, we can simplify that check
                // into whether we are lane 31.
                if (laneIndex == 31)
                {
                    threadNeedleValue = previousWarpLastNeedlePixelValue;
                    threadHaystackValue = previousWarpLastHaystackPixelValue;
                }
                else
                {
                    threadNeedleValue = currentNeedlePixelValue;
                    threadHaystackValue = currentHaystackPixelValue;
                }

                // Exchange "previous pixel" values through shuffle instructions
                radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);
                radialIntersectionCountImagePixelType previousHaystackPixelValue = __shfl_sync(0xFFFFFFFF, threadHaystackValue,
                                                                                               targetThread);

                // The distance measure this function computes is based on deltas between pairs of pixels
                int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
                int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

                // This if statement makes a massive difference in the clutter resistant performance of this method
                // It only counts least squares differences if the needle image has a change in intersection count
                // Which is usually something very specific to that object.
                if (needleDelta != 0)
                {
                    threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
                }

                // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
                previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
                previousWarpLastHaystackPixelValue = currentHaystackPixelValue;
            }
#if ENABLE_RICI_COMPARISON_EARLY_EXIT
            // At the end of each block of 8 rows, check whether we can do an early exit
            // This also works for the constant image
            if (row != (spinImageWidthPixels - 1))
            {
                int intermediateDistance = warpAllReduceSum(threadScore);
                if (intermediateDistance >= distanceToBeat)
                {
                    return intermediateDistance;
                }
            }
#endif
        }

        int imageScore = warpAllReduceSum(threadScore);

        return imageScore;
    }

}

__global__ void ComputePairDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> otherDescriptors,
    int *results)
{
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    int distanceScore;
    int needleSquaredSum = RICIUTILITES::computeImageSquaredSumGPU(descriptors[descriptorIndex]);
    bool needleImageIsConstant = needleSquaredSum == 0;

    if (!needleImageIsConstant)
    {
        distanceScore = RICIUTILITES::compareRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, descriptorIndex);
    }
    else
    {
        distanceScore = RICIUTILITES::compareConstantRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, descriptorIndex);
    }

    if (threadIdx.x == 0)
    {
        results[descriptorIndex] = distanceScore;
    }
}

void DescriptorDistance::RICI::ComputePairWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> haystackDescriptors)
{

    const unsigned int numDescriptors = needleDescriptors.length;

    int *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(int));

    ComputePairDistance<<<numDescriptors, 32>>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    unsigned int exponent = 0;

    for (unsigned int i = 0; i < numDescriptors; i++)
    {
        while (pow(2, exponent) < results[i])
            exponent++;
        // std::cout << "Distance at " << i << ": " << results[i] << std::endl;
    }

    float min = 0.0f, max = 2048; // pow(2,exponent);//1024.0f;

    std::vector<float> fResults;
    fResults.reserve(numDescriptors);
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back(results[i]);

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/rici/combined_pairwise.json");

    cudaFree(results);
}

__global__ void ComputeCrossDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> otherDescriptors,
    DescriptorDistance::intDistances *results)
{

    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;

    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    int distanceScore;
    int needleSquaredSum = RICIUTILITES::computeImageSquaredSumGPU(descriptors[descriptorIndex]);
    bool needleImageIsConstant = needleSquaredSum == 0;

    if (!needleImageIsConstant)
    {
        distanceScore = RICIUTILITES::compareRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, otherDescriptorIndex);
    }
    else
    {
        distanceScore = RICIUTILITES::compareConstantRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, otherDescriptorIndex);
    }

    if (laneIndex == 0 && blockYZIndex < otherDescriptors.length)
    {
        atomicMin(&results[descriptorIndex].min, distanceScore);
        atomicMax(&results[descriptorIndex].max, distanceScore);
        atomicAdd(&results[descriptorIndex].avg, distanceScore);
    }
}

void DescriptorDistance::RICI::ComputeCrossWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> haystackDescriptors)
{
    const unsigned int numDescriptors = needleDescriptors.length;
    const unsigned int numOtherDescriptors = haystackDescriptors.length;

    intDistances *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(intDistances));

    fillDistances(results, numDescriptors);

    const auto [numY, numZ] = findCorrectDimensions(numOtherDescriptors);

    ComputeCrossDistance<<<dim3(numDescriptors, numY, numZ), 32>>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    unsigned int numZero = 0;

    for (unsigned int i = 0; i < numDescriptors; i++)
    {
        // std::cout << "distances at " << i << ", min:" << results[i].min << ", max:" << results[i].max << ", avg:" << (results[i].avg / numOtherDescriptors) << std::endl;
        // if(results[i].min > (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "below min" << std::endl;
        // if(results[i].max < (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "above max" << std::endl;
        // if(results[i].min == 0)
        //     numZero++;
    }

    // std::cout << "Number of zero distances: " << numZero << std::endl;

    unsigned int exponent = 0;

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        while(pow(2,exponent) < results[i].min || pow(2,exponent) < (results[i].avg / numOtherDescriptors) || pow(2,exponent) < results[i].max)
            exponent++;
        // std::cout << "Distance at " << i << ": " << results[i] << std::endl;
    }

    float min = 0.0f, max = pow(2,exponent);//1024.0f;

    std::vector<float> fResults;
    fResults.reserve(numDescriptors);
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back(results[i].min);

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/combined/histogram_E_min_RICI.json");

    fResults.clear();
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back((results[i].avg / numOtherDescriptors));

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/combined/histogram_E_avg_RICI.json");

    fResults.clear();
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back(results[i].max);

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/combined/histogram_E_max_RICI.json");

    cudaFree(results);
}

__global__ void ComputeCrossDistanceWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> otherDescriptors,
    ShapeDescriptor::gpu::array<float> thresholds,
    unsigned int *results)
{
    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;

    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    int distanceScore;
    int needleSquaredSum = RICIUTILITES::computeImageSquaredSumGPU(descriptors[descriptorIndex]);
    bool needleImageIsConstant = needleSquaredSum == 0;

    if (!needleImageIsConstant)
    {
        distanceScore = RICIUTILITES::compareRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, otherDescriptorIndex);
    }
    else
    {
        distanceScore = RICIUTILITES::compareConstantRadialIntersectionCountImagePairGPU(
            descriptors.content, descriptorIndex,
            otherDescriptors.content, otherDescriptorIndex);
    }

    if (laneIndex == 0 && distanceScore < thresholds[descriptorIndex] && blockYZIndex < otherDescriptors.length)
    {
        atomicAdd(&results[descriptorIndex], 1);
    }
}

void DescriptorDistance::RICI::ComputeCrossWiseWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> haystackDescriptors,
    ShapeDescriptor::gpu::array<float> thresholds)
{
    const unsigned int numDescriptors = needleDescriptors.length;
    const unsigned int numOtherDescriptors = haystackDescriptors.length;

    unsigned int *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(unsigned int));

    fillZeros(results, numDescriptors);

    const auto [numY, numZ] = findCorrectDimensions(numOtherDescriptors);

    ComputeCrossDistanceWithThreshold<<<dim3(numDescriptors, numY, numZ), 32>>>(needleDescriptors, haystackDescriptors, thresholds, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    for (unsigned int i = 0; i < numDescriptors; i++)
    {
        std::cout << "Number of distances below threshold " << i << ": " << results[i] << std::endl;
    }

    cudaFree(results);
}
