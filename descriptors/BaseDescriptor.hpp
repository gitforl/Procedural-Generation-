#pragma once

#include <string>
#include <vector>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>
#include <utilities/mathUtilities.hpp>

class BaseDescriptor {
    protected:
        BaseDescriptor(std::string objSrcPath);

        std::string objSrcPath;
        ShapeDescriptor::cpu::Mesh mesh;
        ShapeDescriptor::gpu::Mesh gpuMesh;
        ShapeDescriptor::cpu::Mesh alteredMesh;
        ShapeDescriptor::gpu::Mesh alteredGpuMesh;
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins;
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins;

        StringUIntMap vertexMap;
        std::vector<float> comparisonValues;
        float averageDistance = 0.0f;
        float standardDeviation = 0.0f;
    
        void InitializeMesh();
        void InitializeAlteredMesh();
        void InitializeDescriptorOrigins();

        virtual void CreateReferenceDescriptors() = 0;
        virtual void CreateAlteredDescriptors() = 0;
        virtual void Compare() = 0;
        virtual void ComputeAverageDistance();
        virtual void ComputeStandardDeviation();
    public:
        virtual void ApplyNoise(float noiseLevel);
        void RunSingleNoiseTest(float noiseLevel);
        void RunNoiseTestVaryingLevels(std::vector<float> noiseLevels);
};