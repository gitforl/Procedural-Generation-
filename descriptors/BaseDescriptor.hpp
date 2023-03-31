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

#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Real_timer.h>
#include <CGAL/tags.h>

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
        void ChopUpMesh();
    public:
        virtual void ApplyNoise(float noiseLevel);
        void RunNoiseTestAtLevel(float noiseLevel);
        void RunNoiseTestAtVaryingLevels(std::vector<float> noiseLevels);
        void MeshSelfIntersects();
        void CGALMeshTest();
};