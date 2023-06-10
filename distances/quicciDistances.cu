#include "../utilities/descriptorDistance.cuh"
#include "utility.cuh"

__inline__ __device__ unsigned int getChunkAt(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex) {
    return image[imageIndex].contents[chunkIndex];
}

__inline__ __device__ int computeImageSumGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t imageIndex) 
{

    const int laneIndex = threadIdx.x % 32;

    unsigned int threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes images are multiples of warp size wide");

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, imageIndex, chunk);
        threadSum += __popc(needleChunk);
    }

    int sum = warpAllReduceSum(threadSum);

    return sum;
}

__device__ __forceinline__ float computeQUICCIThreadDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    const size_t descriptorIndex,
    const size_t otherDescriptorsIndex
)
{
    const int laneIndex = threadIdx.x;

    int referenceImageBitCount = computeImageSumGPU(descriptors.content, descriptorIndex);
    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

    bool needleImageIsConstant = referenceImageBitCount == 0;

    float threadWeightedHammingDistance = 0;

    auto chunk = laneIndex;

    if(!needleImageIsConstant) 
    {

        unsigned int needleChunk = getChunkAt(descriptors.content, descriptorIndex, chunk);
        unsigned int haystackChunk = getChunkAt(otherDescriptors.content, otherDescriptorsIndex, chunk);

        threadWeightedHammingDistance += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);

    } else 
    {
        unsigned int haystackChunk = getChunkAt(otherDescriptors.content, descriptorIndex, chunk);

        threadWeightedHammingDistance += float(__popc(haystackChunk));
        
    }

    return threadWeightedHammingDistance;
}

__global__
void ComputePairDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    float *results
)
{
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;


    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, descriptorIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0)
    {
        results[blockIdx.x] = weightedHammingDistance;
    }
}

void DescriptorDistance::QUICCI::ComputePairWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors
)
{

    const unsigned int numDescriptors = needleDescriptors.length;

    float *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(float));

    ComputePairDistance<<<numDescriptors, 32 >>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());


    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        // std::cout << "Distance at" << i << ": " << results[i] << std::endl;
        // if(results[i] < min) min = results[i];
        // if(results[i] > max) max = results[i];
    }


    float min = 0.0f, max = 1024.0f;
    showHistogram(results, numDescriptors, 16, max, min, "../images/json/clutter/histogram_C_4.json");

    cudaFree(results);

}

__global__
void ComputeCrossDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    DescriptorDistance::distances *results
)
{
 
    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;


    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, otherDescriptorIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0 && blockYZIndex < otherDescriptors.length)
    {
        atomicMinFloat(&results[descriptorIndex].min, weightedHammingDistance);
        atomicMaxFloat(&results[descriptorIndex].max, weightedHammingDistance);
        atomicAdd(&results[descriptorIndex].avg, weightedHammingDistance);
    }
}

void DescriptorDistance::QUICCI::ComputeCrossWise(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors
    )
{
    const unsigned int numDescriptors = needleDescriptors.length;
    const unsigned int numOtherDescriptors = haystackDescriptors.length;

    distances *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(distances));

    fillDistances(results, numDescriptors);

    const auto [numY, numZ] = findCorrectDimensions(numOtherDescriptors);

    ComputeCrossDistance<<<dim3(numDescriptors, numY, numZ), 32 >>>(needleDescriptors, haystackDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // unsigned int numZero = 0;

    distances overallDistances = {std::numeric_limits<float>::max(), 0, 0};

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        if(results[i].min < overallDistances.min) overallDistances.min = results[i].min;
        if(results[i].max > overallDistances.max) overallDistances.max = results[i].max;
        overallDistances.avg += (results[i].avg / numOtherDescriptors);


        // std::cout << "distances at " << i << ", min:" << results[i].min << ", max:" << results[i].max << ", avg:" <<  << std::endl;
        // if(results[i].min > (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "below min" << std::endl;
        // if(results[i].max < (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "above max" << std::endl;         
        // if(results[i].min == 0)
        //     numZero++;
    }

    // std::cout << "Number of zero distances: " << numZero << std::endl;

    std::cout << "Min distance: " << overallDistances.min << std::endl;
    std::cout << "Max distance: " << overallDistances.max << std::endl;
    std::cout << "Avg distance: " << (overallDistances.avg / numDescriptors) << std::endl;

    float min = 0.0f, max = 1024.0f;

    std::vector<float> mins, avgs, maxs;
    mins.reserve(numDescriptors);
    // avgs.reserve(numDescriptors);
    // maxs.reserve(numDescriptors);

    for(size_t i = 0; i < numDescriptors; i++)
    {
        mins.emplace_back(results[i].min);
        // avgs.emplace_back(results[i].avg);
        // maxs.emplace_back(results[i].max);
    }

    showHistogram(mins.data(), numDescriptors, 16, max, min, "../images/json/combined_quicci/histogram_I_min_RICI.json");
    // showHistogram(avgs.data(), numDescriptors, 16, max, min, "../images/json/occlusion/histogram_I_avg.json");
    // showHistogram(maxs.data(), numDescriptors, 16, max, min, "../images/json/occlusion/histogram_I_max.json");

    cudaFree(results);
}

__global__
void ComputeCrossDistanceWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    ShapeDescriptor::gpu::array<float> thresholds,
    unsigned int *results
)
{
 
    const size_t blockYZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockYZIndex < otherDescriptors.length ? blockYZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;


    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, otherDescriptorIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0 && weightedHammingDistance < thresholds[descriptorIndex] && blockYZIndex < otherDescriptors.length)
    {
        atomicAdd(&results[descriptorIndex], 1);
    }
}

void DescriptorDistance::QUICCI::ComputeCrossWiseWithThreshold(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors,
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
   

