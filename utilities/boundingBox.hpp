#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <vector>
#include <limits>

namespace BoundingBoxUtilities{

    namespace GeometricTerms
    {
        enum Axis{X = 0, Y = 1, Z = 2};
    };

    struct BoundingBox
    {
        BoundingBox(ShapeDescriptor::cpu::float3 min, ShapeDescriptor::cpu::float3 max);
        BoundingBox(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount);

        ShapeDescriptor::cpu::float3 min, max;

        ShapeDescriptor::cpu::float3 center() const;
        ShapeDescriptor::cpu::float3 span() const;
    };

    struct BoundingBoxNode
    {
        BoundingBoxNode(ShapeDescriptor::cpu::float3 min, ShapeDescriptor::cpu::float3 max);

        BoundingBox boundingBox;
        BoundingBoxNode *left = NULL;
        BoundingBoxNode *right = NULL;

        void SplitBoundingBox(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount, unsigned int n);
    };

    struct BoundingBoxTree
    {
        BoundingBoxTree(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount, unsigned int depth);

        BoundingBoxNode *root;
        ShapeDescriptor::cpu::float3 translation;
        float scale;

        void setTranslation(ShapeDescriptor::cpu::float3 newTranslation);
        void setScale(float newScale);
    };

    struct MinimumVolumeBinarySplit
    {
        MinimumVolumeBinarySplit(ShapeDescriptor::cpu::float3 * vertices, const size_t vertexCount, const BoundingBox &parentBound);

        ShapeDescriptor::cpu::float3 * leftArray;
        ShapeDescriptor::cpu::float3 * rightArray;
        size_t leftVertexCount, rightVertexCount;

        ShapeDescriptor::cpu::float3 leftMin, leftMax;
        ShapeDescriptor::cpu::float3 rightMin, rightMax;

        bool CanSplitFurther();
    };

    struct DistanceUntilBoundingBoxesTouchFinder
    {
        DistanceUntilBoundingBoxesTouchFinder(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree);

        float FindMaxDistance(const ShapeDescriptor::cpu::float3 direction, float previousMax = 0.0f);
        float FindMaxDistanceAccelerated(const ShapeDescriptor::cpu::float3 direction, float previousMax = 0.0f);
        
        private:
            const BoundingBoxTree &movingObjectTree;
            const BoundingBoxTree &stationaryObjectTree;

            ShapeDescriptor::cpu::float3 direction;

            float maxValidDistance;

            inline float ExploreNodePairDistance(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode);
            void ExploreDistanceBetweenNodes(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode);
            void ExploreDistanceBetweenNodesAccelerated(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode);
            float FindMaxDistanceBoxesTouch(const BoundingBox &movingBox, const BoundingBox &stationaryBox);
            bool ModelBoundsTouchAlongAxisUnderMovement(const BoundingBox &movingBox, const BoundingBox &stationaryBox, float distance, GeometricTerms::Axis axis);

            inline ShapeDescriptor::cpu::float3 transformAsMovingObject(const ShapeDescriptor::cpu::float3 &float3, const float distance = 0.0f);
            inline ShapeDescriptor::cpu::float3 transformAsStationaryObject(const ShapeDescriptor::cpu::float3 &float3);
    };

    float FindMaxDistanceUntilBoundsTouch(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction);
    float FindMaxValidDistanceUntilBoundsTouch(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction, float previousValidMax = 0.0f);
    float FindMaxValidDistanceUntilBoundsTouchAccelerated(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction, float previousValidMax = 0.0f);
}