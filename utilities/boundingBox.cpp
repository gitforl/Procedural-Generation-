#include "boundingBox.hpp"


inline ShapeDescriptor::cpu::float3 CreateFloat3WithMaxValues()
{
    float max = std::numeric_limits<float>::max();
    return ShapeDescriptor::cpu::float3(max, max, max);
}

inline ShapeDescriptor::cpu::float3 CreateFloat3WithMinValues()
{
    float min = -std::numeric_limits<float>::max();
    return ShapeDescriptor::cpu::float3(min, min, min);
}

BoundingBoxUtilities::BoundingBox::BoundingBox(ShapeDescriptor::cpu::float3 min, ShapeDescriptor::cpu::float3 max):
min(min),
max(max)
{}

inline void UpdateLeftIfRightGreater(float &left, float &right)
{
    if(left < right) left = right;
}

inline void UpdateLeftIfRightLess(float &left, float &right)
{
    if(left > right) left = right;
}

BoundingBoxUtilities::BoundingBox::BoundingBox(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount){
    for (int i = 0; i < vertexCount; i++){
        UpdateLeftIfRightLess(min.x, vertices[i].x);
        UpdateLeftIfRightLess(min.y, vertices[i].y);
        UpdateLeftIfRightLess(min.z, vertices[i].z);
        UpdateLeftIfRightGreater(max.x, vertices[i].x);
        UpdateLeftIfRightGreater(max.y, vertices[i].y);
        UpdateLeftIfRightGreater(max.z, vertices[i].z);
    }
}

ShapeDescriptor::cpu::float3 BoundingBoxUtilities::BoundingBox::center() const{
    ShapeDescriptor::cpu::float3 center = (min + max) / 2;
    return center;
}

ShapeDescriptor::cpu::float3 BoundingBoxUtilities::BoundingBox::span() const{
    ShapeDescriptor::cpu::float3 span = max - min;
    return span;
}

inline float GetFloat3AxisValue(const ShapeDescriptor::cpu::float3 &float3, const BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    switch (axis)
    {
    case BoundingBoxUtilities::GeometricTerms::Axis::X :
        return float3.x;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Y :
        return float3.y;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Z :
        return float3.z;
        break;
    
    default:
        throw std::out_of_range ("float3 index out of range");
        break;
    }
}

inline void SetFloat3AxisValue(ShapeDescriptor::cpu::float3 &float3, BoundingBoxUtilities::GeometricTerms::Axis axis, float value)
{
    switch (axis)
    {
    case BoundingBoxUtilities::GeometricTerms::Axis::X :
        float3.x = value;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Y :
        float3.y = value;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Z :
        float3.z = value;
        break;
    
    default:
        throw std::out_of_range ("float3 index out of range");
        break;
    }
}

inline bool Float3LessAlongAxis(const ShapeDescriptor::cpu::float3 &target, const ShapeDescriptor::cpu::float3 &other, const BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    return GetFloat3AxisValue(target, axis) < GetFloat3AxisValue(other, axis);
}

inline void UpdateFloat3IfOtherLessAlongAxis(ShapeDescriptor::cpu::float3 &target, ShapeDescriptor::cpu::float3 &other, BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    float currentValue = GetFloat3AxisValue(target, axis);
    float otherValue = GetFloat3AxisValue(other, axis);

    if(otherValue < currentValue) SetFloat3AxisValue(target, axis, otherValue); 
}

inline void UpdateFloat3IfOtherGreaterAlongAxis(ShapeDescriptor::cpu::float3 &target, ShapeDescriptor::cpu::float3 &other, BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    float currentValue = GetFloat3AxisValue(target, axis);
    float otherValue = GetFloat3AxisValue(other, axis);

    if(otherValue > currentValue) SetFloat3AxisValue(target, axis, otherValue); 
}

inline float findCombinedVolume(ShapeDescriptor::cpu::float3 leftMin, ShapeDescriptor::cpu::float3 leftMax, ShapeDescriptor::cpu::float3 rightMin, ShapeDescriptor::cpu::float3 rightMax)
{
    ShapeDescriptor::cpu::float3 leftSpan = leftMax - leftMin;
    ShapeDescriptor::cpu::float3 rightSpan = rightMax - rightMin;

    float leftVolume = leftSpan.x * leftSpan.y * leftSpan.z;
    float rightVolume = rightSpan.x * rightSpan.y * rightSpan.z;

    return leftVolume + rightVolume;
}

inline void swapFloats(float &a, float &b)
{
    float temp = a;
    a = b;
    b = temp;
}

inline float findEquilateralDegree(ShapeDescriptor::cpu::float3 span)
{
    float minValue = span.x;
    float midValue = span.y;
    float maxValue = span.z;

    if(maxValue < minValue)
        swapFloats(minValue, maxValue);
    if(midValue < minValue)
        swapFloats(minValue, midValue);
    if(maxValue < midValue)
        swapFloats(midValue, maxValue);

    float scale1 = midValue / minValue;
    float scale2 = maxValue / midValue;

    return (scale1 + scale2) / 2.0f;
}

BoundingBoxUtilities::BoundingBoxNode::BoundingBoxNode(ShapeDescriptor::cpu::float3 min, ShapeDescriptor::cpu::float3 max):
boundingBox(BoundingBoxUtilities::BoundingBox(min, max))
{}

void BoundingBoxUtilities::BoundingBoxNode::SplitBoundingBox(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount, unsigned int n)
{
    auto minimumVolumeSplit = BoundingBoxUtilities::MinimumVolumeBinarySplit(vertices, vertexCount, boundingBox);
    left = new BoundingBoxUtilities::BoundingBoxNode(minimumVolumeSplit.leftMin, minimumVolumeSplit.leftMax);
    right = new BoundingBoxUtilities::BoundingBoxNode(minimumVolumeSplit.rightMin, minimumVolumeSplit.rightMax);

    if(n > 0 && minimumVolumeSplit.CanSplitFurther())
    {
        left->SplitBoundingBox(minimumVolumeSplit.leftArray, minimumVolumeSplit.leftVertexCount, n - 1);
        right->SplitBoundingBox(minimumVolumeSplit.rightArray, minimumVolumeSplit.rightVertexCount, n - 1);
    }
}

BoundingBoxUtilities::BoundingBoxTree::BoundingBoxTree(ShapeDescriptor::cpu::float3 * vertices, size_t vertexCount, unsigned int depth)
{
    translation = {0.0f, 0.0f, 0.0f};
    scale = 1.0f;

    auto boundingBox = BoundingBox(vertices, vertexCount);

    root = new BoundingBoxNode(boundingBox.min, boundingBox.max);
    if(depth > 0)
        root->SplitBoundingBox(vertices, vertexCount, depth);
}

void BoundingBoxUtilities::BoundingBoxTree::setScale(float newScale)
{
    scale = newScale;
}

void BoundingBoxUtilities::BoundingBoxTree::setTranslation(ShapeDescriptor::cpu::float3 newTranslation)
{
    translation = newTranslation;
}

inline ShapeDescriptor::cpu::float3 ComputeLeftVolumeCenter(const BoundingBoxUtilities::BoundingBox &parentBound, BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    ShapeDescriptor::cpu::float3 center = parentBound.center();
    float leftMin = GetFloat3AxisValue(parentBound.min, axis);
    float leftMax = GetFloat3AxisValue(center, axis);
    float leftCenter = (leftMin + leftMax) / 2;
    SetFloat3AxisValue(center, axis, leftCenter);

    return center;
}

inline ShapeDescriptor::cpu::float3 ComputeRightVolumeCenter(const BoundingBoxUtilities::BoundingBox &parentBound, BoundingBoxUtilities::GeometricTerms::Axis axis)
{
    ShapeDescriptor::cpu::float3 center = parentBound.center();
    float rightMax = GetFloat3AxisValue(parentBound.max, axis);
    float rightMin = GetFloat3AxisValue(center, axis);
    float rightCenter = (rightMin + rightMax) / 2;
    SetFloat3AxisValue(center, axis, rightCenter);

    return center;
}

BoundingBoxUtilities::MinimumVolumeBinarySplit::MinimumVolumeBinarySplit(ShapeDescriptor::cpu::float3 * vertices, const size_t vertexCount, const BoundingBox &parentBound)
{

    // float minimumVolume = std::numeric_limits<float>::max();
    float metric = std::numeric_limits<float>::max();
    const auto parentCenter = parentBound.center();

    std::vector<GeometricTerms::Axis> axes = {GeometricTerms::Axis::X, GeometricTerms::Axis::Y, GeometricTerms::Axis::Z};

    for(auto axis : axes)
    {
        ShapeDescriptor::cpu::float3 verticesDividedAlongCenter[vertexCount];
        ShapeDescriptor::cpu::float3 leftMin = CreateFloat3WithMaxValues();
        ShapeDescriptor::cpu::float3 leftMax = CreateFloat3WithMinValues();
        ShapeDescriptor::cpu::float3 rightMin = CreateFloat3WithMaxValues();
        ShapeDescriptor::cpu::float3 rightMax = CreateFloat3WithMinValues();

        unsigned int index = 0;
        unsigned int reverseIndex = vertexCount - 1;

        for(unsigned int i = 0; i < vertexCount; i++)
        {
            auto vertex = vertices[i];
            if(Float3LessAlongAxis(vertex, parentCenter, axis)){

                for(auto otherAxis : axes)
                {
                    UpdateFloat3IfOtherLessAlongAxis(leftMin, vertex, otherAxis);
                    UpdateFloat3IfOtherGreaterAlongAxis(leftMax, vertex, otherAxis);
                }

                verticesDividedAlongCenter[index] = vertex;
                index++;
            }
            else {
                for(auto otherAxis : axes)
                {
                    UpdateFloat3IfOtherLessAlongAxis(rightMin, vertex, otherAxis);
                    UpdateFloat3IfOtherGreaterAlongAxis(rightMax, vertex, otherAxis);
                }

                verticesDividedAlongCenter[reverseIndex] = vertices[i];
                reverseIndex--;
            }
        }

        auto leftSpan = leftMax - leftMin;
        auto rightSpan = rightMax - rightMin;
        float currentVolume = findCombinedVolume(leftMin, leftMax, rightMin, rightMax);
        float currentMetric = currentVolume * findEquilateralDegree(leftSpan) * findEquilateralDegree(rightSpan);

        if(currentMetric < metric)
        {
            for(unsigned int i = 0; i < vertexCount; i++)
            {
                vertices[i] = verticesDividedAlongCenter[i];
            }

            MinimumVolumeBinarySplit::leftArray = &vertices[0];
            MinimumVolumeBinarySplit::rightArray = &vertices[reverseIndex + 1];

            MinimumVolumeBinarySplit::leftVertexCount = index;
            MinimumVolumeBinarySplit::rightVertexCount = vertexCount - index;

            MinimumVolumeBinarySplit::leftMin = leftMin;
            MinimumVolumeBinarySplit::leftMax = leftMax;
            MinimumVolumeBinarySplit::rightMin = rightMin;
            MinimumVolumeBinarySplit::rightMax = rightMax;
            // minimumVolume = currentVolume;
            metric = currentMetric;
        }
    }

}

bool BoundingBoxUtilities::MinimumVolumeBinarySplit::CanSplitFurther()
{
    return (leftVertexCount > 1 && rightVertexCount > 1);
}

float BoundingBoxUtilities::FindMaxDistanceUntilBoundsTouch(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction)
{
    auto boundsTouchFinder = DistanceUntilBoundingBoxesTouchFinder(movingObjectTree, stationaryObjectTree);
    float distance = boundsTouchFinder.FindMaxDistance(direction);
    return distance;
}

float BoundingBoxUtilities::FindMaxValidDistanceUntilBoundsTouch(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction, float previousValidMax)
{
    auto boundsTouchFinder = DistanceUntilBoundingBoxesTouchFinder(movingObjectTree, stationaryObjectTree);
    float distance = boundsTouchFinder.FindMaxDistance(direction, previousValidMax);
    return distance;
}

float BoundingBoxUtilities::FindMaxValidDistanceUntilBoundsTouchAccelerated(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree, const ShapeDescriptor::cpu::float3 direction, float previousValidMax)
{
    auto boundsTouchFinder = DistanceUntilBoundingBoxesTouchFinder(movingObjectTree, stationaryObjectTree);
    float distance = boundsTouchFinder.FindMaxDistanceAccelerated(direction, previousValidMax);
    return distance;
}

BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::DistanceUntilBoundingBoxesTouchFinder(const BoundingBoxTree &movingObjectTree, const BoundingBoxTree &stationaryObjectTree):
movingObjectTree(movingObjectTree),
stationaryObjectTree(stationaryObjectTree)
{}

ShapeDescriptor::cpu::float3 operator/ (ShapeDescriptor::cpu::float3 dividend, ShapeDescriptor::cpu::float3 divisor) {
    float3 out;
    out.x = dividend.x / divisor.x;
    out.y = dividend.y / divisor.y;
    out.z = dividend.z / divisor.z;
    return out;
}

inline ShapeDescriptor::cpu::float2 GetWithoutComponent(const ShapeDescriptor::cpu::float3 &float3, BoundingBoxUtilities::GeometricTerms::Axis component)
{
    switch (component)
    {
    case BoundingBoxUtilities::GeometricTerms::Axis::X:
        return {float3.y, float3.z};
        break;
    case BoundingBoxUtilities::GeometricTerms::Axis::Y:
        return {float3.x, float3.z};
        break;
    case BoundingBoxUtilities::GeometricTerms::Axis::Z:
        return {float3.x, float3.y};
        break;
    
    default:
        throw std::out_of_range ("Invalid Axis");
        break;
    }
}

inline float GetFloat3Component(const ShapeDescriptor::cpu::float3 &float3, const BoundingBoxUtilities::GeometricTerms::Axis component)
{
    switch (component)
    {
    case BoundingBoxUtilities::GeometricTerms::Axis::X :
        return float3.x;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Y :
        return float3.y;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Z :
        return float3.z;
        break;
    
    default:
        throw std::out_of_range ("float3 index out of range");
        break;
    }
}

inline bool RangesOverlap(float start0, float end0, float start1, float end1)
{
    return !(end0 < start1 || end1 < start0);
}

inline bool RectangleOverlap(ShapeDescriptor::cpu::float2 &start0, ShapeDescriptor::cpu::float2 &end0, ShapeDescriptor::cpu::float2 &start1, ShapeDescriptor::cpu::float2 &end1)
{
    return RangesOverlap(start0.x, end0.x, start1.x, end1.x) && RangesOverlap(start0.y, end0.y, start1.y, end1.y);
}

bool BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::ModelBoundsTouchAlongAxisUnderMovement(const BoundingBox &movingBox, const BoundingBox &stationaryBox, float distance, GeometricTerms::Axis axis)
{
    ShapeDescriptor::cpu::float2 stationaryMin = GetWithoutComponent(transformAsStationaryObject(stationaryBox.min), axis);
    ShapeDescriptor::cpu::float2 stationaryMax = GetWithoutComponent(transformAsStationaryObject(stationaryBox.max), axis);

    ShapeDescriptor::cpu::float2 movingMin = GetWithoutComponent(transformAsMovingObject(movingBox.min, distance), axis);
    ShapeDescriptor::cpu::float2 movingMax = GetWithoutComponent(transformAsMovingObject(movingBox.max, distance), axis);

    return RectangleOverlap(stationaryMin, stationaryMax, movingMin, movingMax);
}

float BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::FindMaxDistance(const ShapeDescriptor::cpu::float3 direction, float previousMax)
{
    DistanceUntilBoundingBoxesTouchFinder::direction = direction;
    maxValidDistance = previousMax;

    ExploreDistanceBetweenNodes(movingObjectTree.root, stationaryObjectTree.root);

    return maxValidDistance;
}

float BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::FindMaxDistanceAccelerated(const ShapeDescriptor::cpu::float3 direction, float previousMax)
{
    DistanceUntilBoundingBoxesTouchFinder::direction = direction;
    maxValidDistance = previousMax;

    ExploreDistanceBetweenNodesAccelerated(movingObjectTree.root, stationaryObjectTree.root);

    return maxValidDistance;
}

inline void sortFloat4(float array[4])
{
    if(array[0]>array[1]) swapFloats(array[0], array[1]);
    if(array[2]>array[3]) swapFloats(array[2], array[3]);
    if(array[0]>array[2]) swapFloats(array[0], array[2]);
    if(array[1]>array[3]) swapFloats(array[1], array[3]);
    if(array[1]>array[2]) swapFloats(array[1], array[2]);
}

inline bool isLeafNode(const BoundingBoxUtilities::BoundingBoxNode *movingBoxNode)
{
    return movingBoxNode->left == NULL;
}

void BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::ExploreDistanceBetweenNodes(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode)
{
    float distance = FindMaxDistanceBoxesTouch(movingBoxNode->boundingBox, stationaryBoxNode->boundingBox);

    if(distance > maxValidDistance)
    {
        if(isLeafNode(movingBoxNode) && isLeafNode(stationaryBoxNode))
            maxValidDistance = distance;
        else if(isLeafNode(movingBoxNode))
        {
            ExploreDistanceBetweenNodes(movingBoxNode, stationaryBoxNode->left);
            ExploreDistanceBetweenNodes(movingBoxNode, stationaryBoxNode->right);
        }
        else if(isLeafNode(stationaryBoxNode))
        {
            ExploreDistanceBetweenNodes(movingBoxNode->left, stationaryBoxNode);
            ExploreDistanceBetweenNodes(movingBoxNode->right, stationaryBoxNode);
        }
        else
        {
            ExploreDistanceBetweenNodes(movingBoxNode->left, stationaryBoxNode->left);
            ExploreDistanceBetweenNodes(movingBoxNode->left, stationaryBoxNode->right);
            ExploreDistanceBetweenNodes(movingBoxNode->right, stationaryBoxNode->left);
            ExploreDistanceBetweenNodes(movingBoxNode->right, stationaryBoxNode->right);
        }            
    }
}

inline float BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::ExploreNodePairDistance(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode)
{
    float distance = FindMaxDistanceBoxesTouch(movingBoxNode->boundingBox, stationaryBoxNode->boundingBox);

    if(distance > maxValidDistance && isLeafNode(movingBoxNode) && isLeafNode(stationaryBoxNode))
        maxValidDistance = distance;
        
    return distance;
}

void BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::ExploreDistanceBetweenNodesAccelerated(const BoundingBoxNode *movingBoxNode, const BoundingBoxNode *stationaryBoxNode)
{

    struct NodePairAndDistance {
        float distance;
        const BoundingBoxNode *movingBoxNode;
        const BoundingBoxNode *stationaryBoxNode;
        inline bool operator< (const NodePairAndDistance &other)
        {
            return distance < other.distance;
        }
        inline bool operator> (const NodePairAndDistance &other)
        {
            return distance > other.distance;
        }
        static void swapTwoPairs(NodePairAndDistance &left, NodePairAndDistance &right)
        {
            const NodePairAndDistance temp = left;
            left = right;
            right = temp;
        }
        static void sortAscendingTwoPairs(NodePairAndDistance &left, NodePairAndDistance &right)
        {
            if(left > right) swapTwoPairs(left, right);
        }
        static void sortAscendingFourPairs(NodePairAndDistance array[4])
        {
            sortAscendingTwoPairs(array[0], array[1]);
            sortAscendingTwoPairs(array[2], array[3]);
            sortAscendingTwoPairs(array[0], array[2]);
            sortAscendingTwoPairs(array[1], array[3]);
            sortAscendingTwoPairs(array[1], array[2]);
        } 
    };


    if(isLeafNode(movingBoxNode) && isLeafNode(stationaryBoxNode))
        return;
    else if(isLeafNode(movingBoxNode))
    {
        float d0 = ExploreNodePairDistance(movingBoxNode, stationaryBoxNode->left);
        float d1 = ExploreNodePairDistance(movingBoxNode, stationaryBoxNode->right);

        NodePairAndDistance pairs[2];
        pairs[0] = {d0, movingBoxNode, stationaryBoxNode->left};
        pairs[1] = {d1, movingBoxNode, stationaryBoxNode->right};

        NodePairAndDistance::sortAscendingTwoPairs(pairs[0], pairs[1]);
        for(unsigned int i = 0; i < 2; i++)
            ExploreDistanceBetweenNodesAccelerated(pairs[i].movingBoxNode, pairs[i].stationaryBoxNode);
    }
    else if(isLeafNode(stationaryBoxNode))
    {
        float d0 = ExploreNodePairDistance(movingBoxNode->left, stationaryBoxNode);
        float d1 = ExploreNodePairDistance(movingBoxNode->right, stationaryBoxNode);

        NodePairAndDistance pairs[2];
        pairs[0] = {d0, movingBoxNode->left, stationaryBoxNode};
        pairs[1] = {d1, movingBoxNode->right, stationaryBoxNode};

        NodePairAndDistance::sortAscendingTwoPairs(pairs[0], pairs[1]);
        for(unsigned int i = 0; i < 2; i++)
            ExploreDistanceBetweenNodesAccelerated(pairs[i].movingBoxNode, pairs[i].stationaryBoxNode);
    }
    else
    {

        float d0 = ExploreNodePairDistance(movingBoxNode->left, stationaryBoxNode->left);
        float d1 = ExploreNodePairDistance(movingBoxNode->left, stationaryBoxNode->right);
        float d2 = ExploreNodePairDistance(movingBoxNode->right, stationaryBoxNode->left);
        float d3 = ExploreNodePairDistance(movingBoxNode->right, stationaryBoxNode->right);

        NodePairAndDistance pairs[4];
        pairs[0] = {d0, movingBoxNode->left, stationaryBoxNode->left};
        pairs[1] = {d1, movingBoxNode->left, stationaryBoxNode->right};
        pairs[2] = {d2, movingBoxNode->right, stationaryBoxNode->left};
        pairs[3] = {d3, movingBoxNode->right, stationaryBoxNode->right};

        NodePairAndDistance::sortAscendingFourPairs(pairs);
        for(unsigned int i = 0; i < 4; i++)
            ExploreDistanceBetweenNodesAccelerated(pairs[i].movingBoxNode, pairs[i].stationaryBoxNode);
    }            
    
}

float BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::FindMaxDistanceBoxesTouch(const BoundingBox &movingBox, const BoundingBox &stationaryBox)
{
    const auto movingMinStationaryMaxDistances = (transformAsStationaryObject(stationaryBox.max) - transformAsMovingObject(movingBox.min)) / direction;
    const auto movingMaxStationaryMinDistances = (transformAsStationaryObject(stationaryBox.min) - transformAsMovingObject(movingBox.max)) / direction;

    const std::vector<ShapeDescriptor::cpu::float3> oppositeSidesDistances = {movingMinStationaryMaxDistances, movingMaxStationaryMinDistances};
    const std::vector<GeometricTerms::Axis> axes = {GeometricTerms::Axis::X, GeometricTerms::Axis::Y, GeometricTerms::Axis::Z};

    float maxDistance = 0.0f;

    for(auto distances : oppositeSidesDistances)
        for(auto axis : axes)
        {
            const float distance = GetFloat3Component(distances, axis);
            if(distance > maxDistance && ModelBoundsTouchAlongAxisUnderMovement(movingBox, stationaryBox, distance, axis))
                maxDistance = distance;
        }

    return maxDistance;
}

inline ShapeDescriptor::cpu::float3 BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::transformAsMovingObject(const ShapeDescriptor::cpu::float3 &float3, const float distance)
{
    auto scaledPoint = float3 * movingObjectTree.scale;
    auto modelSpaceTranslatedPoint = scaledPoint + movingObjectTree.translation;
    auto movement = distance * direction;
    auto movementTranslatedPoint = modelSpaceTranslatedPoint + movement;
    return movementTranslatedPoint;
}

inline ShapeDescriptor::cpu::float3 BoundingBoxUtilities::DistanceUntilBoundingBoxesTouchFinder::transformAsStationaryObject(const ShapeDescriptor::cpu::float3 &float3)
{
    auto scaledPoint = float3 * stationaryObjectTree.scale;
    auto modelSpaceTranslatedPoint = scaledPoint + stationaryObjectTree.translation;
    return modelSpaceTranslatedPoint;
}