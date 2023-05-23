#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#endif


#ifndef __CUDACC__
    unsigned int __popc(unsigned int x);
#endif

#include <bitset>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <utilities/descriptorDistance.hpp>

#include <shapeDescriptor/utilities/weightedHamming.cuh>

#include <chrono>

#include <utilities/aliases.hpp>

namespace DescriptorDistance {

    struct distances {
        float min, max, avg;
    };

    namespace Hamming {

        void NumberOfDistancesToRandomDesciptorsLowerThanTrueDistance(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> correspondingDescriptors,
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> randomDescriptors
            );

        __global__
        void ComputeCUDA(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors,
            const unsigned int numNeedleDescriptors,
            const unsigned int numHaystackDescriptors,
            float *results
            );
        void ComputeCUDAWrapper(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors,
            const unsigned int numNeedleDescriptors,
            const unsigned int numHaystackDescriptors
            );

        void FindMinDistance(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors
        );

        __global__
        void FindMinDistanceCuda(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
            float *results
        );

        void FindDistances(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors
        );

        __global__
        void FindDistancesCuda(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
            distances *results
        );

        void FindElementWiseDistances(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors, 
            ShapeDescriptor::gpu::array<IndexPair> pairs
        );


        __global__
        void FindElementWiseDistancesCuda(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors, 
            ShapeDescriptor::gpu::array<IndexPair> pairs,
            float *results
        );

        __global__
        void ComputeElementWiseCuda(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors,
            const unsigned int numDescriptors,
            float *results
            );
        void ComputeElementWiseCUDAWrapper(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> otherDescriptors
            );

        __global__
        void CudaComputeDistances(
            ShapeDescriptor::QUICCIDescriptor needle, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            const unsigned int numDescriptors,
            float *result
            );

        void CudaComputeDistancesWrapper(
            ShapeDescriptor::QUICCIDescriptor needle, 
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors, 
            const unsigned int numDescriptors
            );

        void CompareDescriptors(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> correspondingDescriptors,
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> randomDescriptors
            );

        __global__
        void CudaCompareDescriptors(
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors,
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> randomDescriptors,
            float *trueDescriptorDistance,
            unsigned int *results
            );

        void testWrapper();
        __global__
        void test(int *n);

    }
}