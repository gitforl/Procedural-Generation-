#include "../utilities/descriptorDistance.cuh"
#include "utility.cuh"

namespace SIUTILITIES {

    __device__ float computeSpinImagePairCorrelationGPU(
            ShapeDescriptor::SpinImageDescriptor* descriptors,
            ShapeDescriptor::SpinImageDescriptor* otherDescriptors,
            size_t spinImageIndex,
            size_t otherImageIndex,
            float averageX, float averageY) {

        float threadSquaredSumX = 0;
        float threadSquaredSumY = 0;
        float threadMultiplicativeSum = 0;

        spinImagePixelType pixelValueX;
        spinImagePixelType pixelValueY;

        for (int y = 0; y < spinImageWidthPixels; y++)
        {
            const int warpSize = 32;
            for (int x = threadIdx.x % 32; x < spinImageWidthPixels; x += warpSize)
            {
                pixelValueX = descriptors[spinImageIndex].contents[y * spinImageWidthPixels + x];
                pixelValueY = otherDescriptors[otherImageIndex].contents[y * spinImageWidthPixels + x];

                float deltaX = float(pixelValueX) - averageX;
                float deltaY = float(pixelValueY) - averageY;

                threadSquaredSumX += deltaX * deltaX;
                threadSquaredSumY += deltaY * deltaY;
                threadMultiplicativeSum += deltaX * deltaY;
            }
        }

        float squaredSumX = float(sqrt(warpAllReduceSum(threadSquaredSumX)));
        float squaredSumY = float(sqrt(warpAllReduceSum(threadSquaredSumY)));
        float multiplicativeSum = warpAllReduceSum(threadMultiplicativeSum);

        float correlation = multiplicativeSum / (squaredSumX * squaredSumY);

        return correlation;
    }
}

__global__
void ComputePairDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> otherDescriptors,
    int *results
)
{
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");


    float threadSquaredSum = 0;

    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        spinImagePixelType descriptorPixelValue = descriptors[descriptorIndex].contents[i];
        spinImagePixelType correspondingPixelValue = otherDescriptors[descriptorIndex].contents[i];
        spinImagePixelType pixelDelta = descriptorPixelValue - correspondingPixelValue;
        threadSquaredSum += pixelDelta * pixelDelta;
    }

    float totalSquaredSum = warpAllReduceSum(threadSquaredSum);

    if(threadIdx.x == 0) {
        results[descriptorIndex] = totalSquaredSum;
    }
}

void DescriptorDistance::SI::ComputePairWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> haystackDescriptors
)
{

    const unsigned int numDescriptors = needleDescriptors.length;

    int *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(int));

    ComputePairDistance<<<numDescriptors, 32 >>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        std::cout << "Distance at " << i << ": " << results[i] << std::endl;
    }

        unsigned int exponent = 0;

    for (unsigned int i = 0; i < numDescriptors; i++)
    {
        while (pow(2, exponent) < results[i])
            exponent++;
        // std::cout << "Distance at " << i << ": " << results[i] << std::endl;
    }

    float min = 0.0f, max = pow(2,exponent);//1024.0f;

    std::vector<float> fResults;
    fResults.reserve(numDescriptors);
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back(results[i]);

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/si/pairwise_combined.json");

    cudaFree(results);

}


__global__
void ComputeCrossDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> otherDescriptors,
    DescriptorDistance::intDistances *results
)
{
 
    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");


    float threadSquaredSum = 0;

    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        spinImagePixelType descriptorPixelValue = descriptors[descriptorIndex].contents[i];
        spinImagePixelType correspondingPixelValue = otherDescriptors[otherDescriptorIndex].contents[i];
        spinImagePixelType pixelDelta = descriptorPixelValue - correspondingPixelValue;
        threadSquaredSum += pixelDelta * pixelDelta;
    }

    float totalSquaredSum = warpAllReduceSum(threadSquaredSum);

    if(laneIndex == 0 && blockYZIndex < otherDescriptors.length)
    {
        atomicMin(&results[descriptorIndex].min, totalSquaredSum);
        atomicMax(&results[descriptorIndex].max, totalSquaredSum);
        atomicAdd(&results[descriptorIndex].avg, totalSquaredSum);
    }
}

void DescriptorDistance::SI::ComputeCrossWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> haystackDescriptors
    )
{
    const unsigned int numDescriptors = needleDescriptors.length;
    const unsigned int numOtherDescriptors = haystackDescriptors.length;

    intDistances *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(intDistances));

    fillDistances(results, numDescriptors);

    const auto [numY, numZ] = findCorrectDimensions(numOtherDescriptors);

    ComputeCrossDistance<<<dim3(numDescriptors, numY, numZ), 32 >>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

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

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/si/I_min.json");

    fResults.clear();
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back((results[i].avg / numOtherDescriptors));

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/si/I_avg.json");

    fResults.clear();
    for (size_t i = 0; i < numDescriptors; i++)
        fResults.emplace_back(results[i].max);

    showHistogram(fResults.data(), numDescriptors, 16, max, min, "../images/json/si/I_max.json");
    
    cudaFree(results);
}

__global__
void ComputeCrossDistanceWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> otherDescriptors,
    ShapeDescriptor::gpu::array<float> thresholds,
    unsigned int *results
)
{
    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");


    float threadSquaredSum = 0;

    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        spinImagePixelType descriptorPixelValue = descriptors[descriptorIndex].contents[i];
        spinImagePixelType correspondingPixelValue = otherDescriptors[otherDescriptorIndex].contents[i];
        spinImagePixelType pixelDelta = descriptorPixelValue - correspondingPixelValue;
        threadSquaredSum += pixelDelta * pixelDelta;
    }

    float totalSquaredSum = warpAllReduceSum(threadSquaredSum);

    if(laneIndex == 0 && totalSquaredSum < thresholds[descriptorIndex]  && blockYZIndex < otherDescriptors.length)
    {
        atomicAdd(&results[descriptorIndex], 1);
    }
}

void DescriptorDistance::SI::ComputeCrossWiseWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> haystackDescriptors,
    ShapeDescriptor::gpu::array<float> thresholds
)
{
    const unsigned int numDescriptors = needleDescriptors.length;
    const unsigned int numOtherDescriptors = haystackDescriptors.length;

    unsigned int *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(unsigned int));

    fillZeros(results, numDescriptors);

    const auto [numY, numZ] = findCorrectDimensions(numOtherDescriptors);

    ComputeCrossDistanceWithThreshold<<<dim3(numDescriptors, numY, numZ), 32 >>>(needleDescriptors, haystackDescriptors, thresholds, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        std::cout << "Number of distances below threshold " << i << ": " << results[i] << std::endl;
    }

    
    cudaFree(results);
}
   

