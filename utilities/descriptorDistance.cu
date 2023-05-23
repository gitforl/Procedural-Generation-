#include "descriptorDistance.cuh"

//taken from https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
            __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

//taken from https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__inline__ __device__ unsigned int warpAllReduceSum(unsigned int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ unsigned int getChunkAt(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex) {
    return image[imageIndex].contents[chunkIndex];
}

__inline__ __device__ int computeImageSumGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t imageIndex) {

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


inline __device__ unsigned int CountBitsTrueLeftFalseRight1(const unsigned int left, const unsigned int right)
{
    return __popc(left & !right);
}

inline __device__ unsigned int CountBitsFalseLeftTrueRight1(const unsigned int left, const unsigned int right)
{
    return __popc(!left & right);
}

inline __device__ unsigned int CountBitsTrueInChunk(const unsigned int chunk)
{
    return __popc(chunk);
}

inline unsigned int GetQUICCIChunk1(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex)
{
    return image[imageIndex].contents[chunkIndex];
}

void DescriptorDistance::Hamming::CudaComputeDistancesWrapper(
    ShapeDescriptor::QUICCIDescriptor needle, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    const unsigned int numDescriptors
    )
{

    float * result;
    
    auto start = std::chrono::high_resolution_clock::now();

    cudaMallocManaged(&result, numDescriptors * sizeof(float));

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Malloc duration: " << duration.count() << std::endl;


    start = std::chrono::high_resolution_clock::now();

    CudaComputeDistances<<<numDescriptors, 32>>>(needle, descriptors, numDescriptors, result);


    now = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Run Computation: " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    now = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Sync duration: " << duration.count() << std::endl;


    std::cout << "Result: " << (*result / float(numDescriptors)) << std::endl;

    cudaFree(result);
    
}

__global__
void DescriptorDistance::Hamming::CudaComputeDistances(
    ShapeDescriptor::QUICCIDescriptor needle, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    const unsigned int numDescriptors,
    float *result
    )
{
    
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;

    unsigned int referenceImageBitCount = CountBitsTrueInChunk(needle.contents[laneIndex]);

    unsigned int totalBitsInBitString = spinImageWidthPixels * spinImageWidthPixels;
    unsigned int queryImageUnsetBitCount = totalBitsInBitString - referenceImageBitCount;

    // If any count is 0, bump it up to 1

    #ifdef __CUDACC__
        referenceImageBitCount = max(referenceImageBitCount, 1);
        queryImageUnsetBitCount = max(queryImageUnsetBitCount, 1);
    #else
        referenceImageBitCount = std::max<unsigned int>(referenceImageBitCount, 1);
        queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);
    #endif

    // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
    float missedSetBitPenalty = float(totalBitsInBitString) / float(referenceImageBitCount);
    float missedUnsetBitPenalty = float(totalBitsInBitString) / float(queryImageUnsetBitCount);

    unsigned int needleChunk = needle.contents[laneIndex];
    unsigned int haystackChunk = descriptors[descriptorIndex].contents[laneIndex];

    const unsigned int numberOfBitsTrueInNeedleFalseInHaystack = CountBitsTrueLeftFalseRight1(needleChunk, haystackChunk);
    const unsigned int numberOfBitsFalseInNeedleTrueInHaystack = CountBitsFalseLeftTrueRight1(needleChunk, haystackChunk);

    const float distance = float(numberOfBitsTrueInNeedleFalseInHaystack) * missedSetBitPenalty
                            + float(numberOfBitsFalseInNeedleTrueInHaystack) * missedUnsetBitPenalty;

    atomicAdd(result, distance);
    // results[descriptorIndex] = distance;   
}


void DescriptorDistance::Hamming::NumberOfDistancesToRandomDesciptorsLowerThanTrueDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> correspondingDescriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> randomDescriptors
    )
{

    const unsigned int numDescriptors = (descriptors.length < correspondingDescriptors.length) ? descriptors.length : correspondingDescriptors.length;
    const unsigned int numRandomDescriptors = randomDescriptors.length;

    float *trueDescriptorDistance;
    cudaMallocManaged(&trueDescriptorDistance, numDescriptors * sizeof(float));

    ComputeElementWiseCuda<<<numDescriptors, 32>>>(descriptors, correspondingDescriptors, numDescriptors, trueDescriptorDistance);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    unsigned int *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(unsigned int));

    for(unsigned int i = 0; i < numDescriptors; i++)
        results[i] = 0;

    const unsigned int numZ = numRandomDescriptors / 65535;
    const unsigned int numY = numRandomDescriptors % 65535;

    CudaCompareDescriptors<<<dim3(numDescriptors, numY, numZ), 32 >>>(descriptors, randomDescriptors, trueDescriptorDistance, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaFree(trueDescriptorDistance);

    // std::cout << "num wrond: " << numDescriptors << std::endl;
    unsigned int numWrong = 0;
    float avgWrong = 0.0f;
    for(unsigned int i = 0; i < numDescriptors; i++)
    {

        if(results[i] != 0)
        {
            std::cout << "desc i: " << i << ", distance to beat: " << trueDescriptorDistance[i] << ", val: " << results[i] << std::endl;
            numWrong++;
            avgWrong += float(results[i]) / numDescriptors;
        }
    }

    std::cout << "num wrong: " << numWrong << std::endl;
    std::cout << "avg wrong: " << avgWrong << std::endl;

    cudaFree(results);

}


void DescriptorDistance::Hamming::FindMinDistance(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors
)
{
    
    const unsigned int numDescriptors = descriptors.length;
    const unsigned int numOtherDescriptors = otherDescriptors.length;

    float *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(float));

    for(unsigned int i = 0; i < numDescriptors; i++)
        results[i] = 1024.0f;

    const unsigned int numZ = numOtherDescriptors / 65535;
    const unsigned int numY = numOtherDescriptors % 65535;

    FindMinDistanceCuda<<<dim3(numDescriptors, numY, numZ), 32 >>>(descriptors, otherDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    for(unsigned int i = 0; i < numDescriptors; i++)
        std::cout << results[i] << std::endl;
    
    cudaFree(results);
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
void DescriptorDistance::Hamming::FindMinDistanceCuda(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    float *results
)
{
    
    const size_t blockXZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t randomDescriptorsIndex = blockXZIndex < otherDescriptors.length ? blockXZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;

    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, randomDescriptorsIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0)
        atomicMinFloat(&results[descriptorIndex], weightedHammingDistance);

}


void DescriptorDistance::Hamming::FindDistances(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors
)
{
    
    const unsigned int numDescriptors = descriptors.length;
    const unsigned int numOtherDescriptors = otherDescriptors.length;

    distances *results;
    cudaMallocManaged(&results, numDescriptors * sizeof(distances));

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        results[i] = {1024.0f, 0.0f, 0.0f};
    }

    unsigned int numZ = 1;
    unsigned int numY = numOtherDescriptors;

    if(numOtherDescriptors > 65535)
    {
        float numSquareRoot = sqrt(numOtherDescriptors);
        int flooredRoot = int(numSquareRoot);


        numY = flooredRoot;
        numZ = flooredRoot + 1;

        numY += numY * numZ < numOtherDescriptors ? 1 : 0; 
    }
    

    FindDistancesCuda<<<dim3(numDescriptors, numY, numZ), 32 >>>(descriptors, otherDescriptors, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    unsigned int numZero = 0;

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        // std::cout << "distances at " << i << ", min:" << results[i].min << ", max:" << results[i].max << ", avg:" << (results[i].avg / numOtherDescriptors) << std::endl;
        // if(results[i].min > (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "below min" << std::endl;
        // if(results[i].max < (results[i].avg / numOtherDescriptors))
        //     std::cout << "avg distance at " << i << "above max" << std::endl;         
        if(results[i].min == 0)
            numZero++;
    }

    std::cout << "Number of zero distances: " << numZero << std::endl;
    
    cudaFree(results);
}

__global__
void DescriptorDistance::Hamming::FindDistancesCuda(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    distances *results
)
{
 
    const size_t blockXZIndex = blockIdx.y + blockIdx.z * gridDim.z;
    
    const size_t descriptorIndex = blockIdx.x;
    const size_t otherDescriptorIndex = blockXZIndex < otherDescriptors.length ? blockXZIndex : otherDescriptors.length - 1;
    const int laneIndex = threadIdx.x;


    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, otherDescriptorIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0 && blockXZIndex < otherDescriptors.length)
    {
        atomicMinFloat(&results[descriptorIndex].min, weightedHammingDistance);
        atomicMaxFloat(&results[descriptorIndex].max, weightedHammingDistance);
        atomicAdd(&results[descriptorIndex].avg, weightedHammingDistance);
    }
}


void DescriptorDistance::Hamming::FindElementWiseDistances(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors, 
    ShapeDescriptor::gpu::array<IndexPair> pairs
)
{

    const unsigned int numPairs = pairs.length;

    float *results;
    cudaMallocManaged(&results, numPairs * sizeof(float));


    FindElementWiseDistancesCuda<<<numPairs, 32 >>>(descriptors, otherDescriptors, pairs, results);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    unsigned int numZero = 0;

    for(unsigned int i = 0; i < numPairs; i++)
    {
        std::cout << "distances at " << i << ": " << results[i] << std::endl;   
        if(results[i] == 0)
            numZero++;
    }
    
    std::cout << numZero << std::endl;

    cudaFree(results);

}


__global__
void DescriptorDistance::Hamming::FindElementWiseDistancesCuda(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors, 
    ShapeDescriptor::gpu::array<IndexPair> pairs,
    float *results
)
{
    const size_t descriptorIndex = pairs[blockIdx.x].left;
    const size_t otherDescriptorIndex = pairs[blockIdx.x].right;
    const int laneIndex = threadIdx.x;

    // printf("pair: %d, %d", static_cast<int>(descriptorIndex), static_cast<int>(otherDescriptorIndex));

    float threadWeightedHammingDistance = computeQUICCIThreadDistance(descriptors, otherDescriptors, descriptorIndex, otherDescriptorIndex);

    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0)
        results[descriptorIndex] = weightedHammingDistance;
    

}

__global__ void DescriptorDistance::Hamming::CudaCompareDescriptors(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> randomDescriptors,
    float *trueDescriptorDistance,
    unsigned int *results
    )
{

    const size_t descriptorIndex = blockIdx.x;
    const size_t randomDescriptorsIndex = blockIdx.y + blockIdx.z * 65535;
    const int laneIndex = threadIdx.x;


    int referenceImageBitCount = computeImageSumGPU(descriptors.content, descriptorIndex);
    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

    bool needleImageIsConstant = referenceImageBitCount == 0;

    float threadWeightedHammingDistance = 0;

    auto chunk = laneIndex;

    if(!needleImageIsConstant) 
    {

        unsigned int needleChunk = getChunkAt(descriptors.content, descriptorIndex, chunk);
        unsigned int haystackChunk = getChunkAt(randomDescriptors.content, randomDescriptorsIndex, chunk);

        threadWeightedHammingDistance += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);

    } else 
    {
        unsigned int haystackChunk = getChunkAt(randomDescriptors.content, descriptorIndex, chunk);

        threadWeightedHammingDistance += float(__popc(haystackChunk));
        
    }
    __syncthreads();

    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0 && weightedHammingDistance < trueDescriptorDistance[descriptorIndex])
        atomicAdd(&results[descriptorIndex], 1);

}

void DescriptorDistance::Hamming::ComputeElementWiseCUDAWrapper(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors    
    )
{

    float * results;
    
    auto start = std::chrono::high_resolution_clock::now();

    const unsigned int numDescriptors = descriptors.length;
    cudaMallocManaged(&results, numDescriptors * sizeof(float));

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Malloc duration: " << duration.count() << std::endl;


    start = std::chrono::high_resolution_clock::now();

    ComputeElementWiseCuda<<<numDescriptors, 32>>>(descriptors, otherDescriptors, numDescriptors, results);


    now = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Run Computation: " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    now = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Cuda Sync duration: " << duration.count() << std::endl;

    // unsigned int indexOfMax = 0;
    // float currentMax = 0.0f;

    for(unsigned int i = 0; i < numDescriptors; i++)
    {
        std::cout << "distance at " << i << ": " << results[i] << std::endl;
        // if(results[i] > currentMax) indexOfMax = i;
    }

    cudaFree(results);
    
    // std::cout << indexOfMax << std::endl;
}

__global__
void DescriptorDistance::Hamming::ComputeElementWiseCuda(
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
    const unsigned int numDescriptors,
    float *results
    )
{
    
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;

    int referenceImageBitCount = computeImageSumGPU(descriptors.content, descriptorIndex);
    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);


    bool needleImageIsConstant = referenceImageBitCount == 0;

    float threadWeightedHammingDistance = 0;

    auto chunk = laneIndex;

    if(!needleImageIsConstant) 
    {
        unsigned int needleChunk = getChunkAt(descriptors.content, descriptorIndex, chunk);
        unsigned int haystackChunk = getChunkAt(otherDescriptors.content, descriptorIndex, chunk);

        threadWeightedHammingDistance += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
    } else 
    {
        unsigned int haystackChunk = getChunkAt(otherDescriptors.content, descriptorIndex, chunk);

        threadWeightedHammingDistance += float(__popc(haystackChunk));
        
    }


    float weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);

    if(laneIndex == 0)
        results[descriptorIndex] = weightedHammingDistance;
    
}

void DescriptorDistance::Hamming::testWrapper()
{
    int *n;
    cudaMallocManaged(&n, sizeof(int));


    // for(unsigned int i = 0; i < 32; i++)
    //     n[i] = 0;

    test<<<dim3(32,32),32>>>(n);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // for(unsigned int i = 0; i < 32; i++)
        std::cout << "n: " << n[0] << std::endl;

    cudaFree(n);
}

__global__ void DescriptorDistance::Hamming::test(int *n)
{
    const int laneIndex = threadIdx.x;
    const int xIdx = blockIdx.x;
    const int yIdx = blockIdx.y;

    __shared__ int blockSum;

    if(laneIndex == 0)
        blockSum = 0;
    __syncthreads();

    atomicAdd(&blockSum, xIdx);
    __syncthreads();


    if(laneIndex == 0)
        printf("block (%d , %d): %d\n", xIdx, yIdx, (blockSum / 32));
    // n[laneIndex] = laneIndex;
    
}