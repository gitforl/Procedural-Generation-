#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#endif


#ifndef __CUDACC__
    unsigned int __popc(unsigned int x);
#endif

#include <bitset>
#include <tuple>
#include <vector>
#include <iostream>

#include <json/json.hpp>

using json = nlohmann::json;

namespace DescriptorDistance {

    struct distances {
        float min, max, avg;
    };

    struct intDistances {
        int min, max, avg;
    };
}


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

__inline__ __device__ unsigned int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ unsigned int warpAllReduceSum(size_t val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

inline void fillDistances(DescriptorDistance::distances *results, size_t count)
{
    for(unsigned int i = 0; i < count; i++)
    {
        results[i] = {1024.0f, 0.0f, 0.0f};
    }
}

inline void fillDistances(DescriptorDistance::intDistances *results, size_t count)
{
    for(unsigned int i = 0; i < count; i++)
    {
        results[i] = {std::numeric_limits<int>::max(), 0, 0};
    }
}

inline void fillZeros(unsigned int *results, size_t count)
{
    for(unsigned int i = 0; i < count; i++)
    {
        results[i] = 0;
    }
}

inline std::tuple<size_t, size_t> findCorrectDimensions(size_t count)
{

    size_t numY = count;
    size_t numZ = 1;

    if(count > 65535)
    {
        float numSquareRoot = sqrt(count);
        int flooredRoot = int(numSquareRoot);


        numY = flooredRoot + 1;
        numZ = flooredRoot;

        numZ += numY * numZ < count ? 1 : 0; 
    }

    return {numY, numZ};
}


void showHistogram(float *results, size_t count, size_t rangeCount, float max, float min, std::string outputDir = "../images/json/histogram.json");

