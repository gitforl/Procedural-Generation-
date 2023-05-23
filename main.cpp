#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <filesystem>

#include <shapeDescriptor/utilities/read/MeshLoadUtils.h>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <shapeDescriptor/utilities/spinOriginsGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageSearcher.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>

#include <shapeDescriptor/utilities/kernels/gpuMeshSampler.cuh>

#include <utilities/aliases.hpp>
#include <utilities/meshFunctions.hpp>
#include <utilities/boundingBox.hpp>
#include <descriptors/BaseDescriptor.hpp>
#include <descriptors/QUICCIDescriptor.hpp>
#include <descriptors/RICIDescriptorTester.hpp>
#include <descriptors/SIDescriptorTester.hpp>
#include <descriptors/FPFHDescriptorTester.hpp>

#include <meshModifier/meshModifier.hpp>
#include <openglHandler/openglHandler.hpp>
#include <openglHandler/openglMesh.hpp>

#include <openglHandler/shader.hpp>
#include <meshModifier/model.hpp>

#include <utilities/cgalMeshFunctions.hpp>
#include <meshModifier/cgalMesh.hpp>

#include <chrono>

#include <utilities/descriptorDistance.hpp>
#include <utilities/descriptorDistance.cuh>

#include <initializer_list>

struct face
{
    // vertices
    ShapeDescriptor::cpu::float3 v0;
    ShapeDescriptor::cpu::float3 v1;
    ShapeDescriptor::cpu::float3 v2;

    ShapeDescriptor::cpu::float3 normal;

    // neighbour faces
    face *n0 = nullptr;
    face *n1 = nullptr;
    face *n2 = nullptr;

    face() = default;

    face(
        ShapeDescriptor::cpu::float3 v0,
        ShapeDescriptor::cpu::float3 v1,
        ShapeDescriptor::cpu::float3 v2,
        ShapeDescriptor::cpu::float3 normal) : v0(v0), v1(v1), v2(v2), normal(normal) {}
};


ShapeDescriptor::cpu::float3 operator/=(ShapeDescriptor::cpu::float3 &target, float &other)
{
    return target / other;
}

template <typename T>
void printVector(std::vector<T> vector, std::string headerText = "")
{
    std::cout << headerText << std::endl;
    for (T element : vector)
    {
        std::cout << element << std::endl;
    }
}

std::vector<float> generateNoiseLevels(int numLevels, float multiplier = 2.0f, bool addZeroRef = true){
    
    std::vector<float> noiseLevels;

    if(addZeroRef)
        noiseLevels.push_back(0.0f);

    float currentDistance = 0.01f;

    for(int i = 0; i < numLevels; i++){
        currentDistance = currentDistance * multiplier;
        noiseLevels.push_back(currentDistance);
    }

    return noiseLevels;
}

void saveMeshCopyWithNoise(const std::string objSrcPath, float noiseMagnitude)
{

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    auto vertexMap = MeshFunctions::MapVertexIndices(&mesh);
    auto averageNormals = MeshFunctions::VertexToAverageNormalMap(mesh, vertexMap);

    MeshFunctions::MoveVerticesAlongAverageNormal(&mesh, vertexMap, noiseMagnitude);

    std::string objSuffix = ".obj";

    const std::filesystem::path outPath = objSrcPath.substr(0, objSrcPath.length() - objSuffix.length()) + "-NoiseRange-" + std::to_string(noiseMagnitude) + objSuffix;
    ShapeDescriptor::dump::mesh(mesh, outPath);
    ShapeDescriptor::free::mesh(mesh);
}


void run3DContextDescriptor()
{
    const std::string objSrcPath = "../objects/T100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins =
        ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);

    ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::internal::sampleMesh(gpuMesh, 1000000, 0);

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuDescriptorOrigins =
        ShapeDescriptor::copy::hostArrayToDevice(spinOrigins);

    gpuDescriptorOrigins.length = 500;

    float supportRadius = 10.0f;

    ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors =
        ShapeDescriptor::gpu::generate3DSCDescriptors(
            pointCloud,
            gpuDescriptorOrigins,
            0.2f,
            0.1f,
            2.5f);

    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> hostDescriptors =
        ShapeDescriptor::copy::deviceArrayToHost(descriptors);

    // ShapeDescriptor::dump::descriptors(hostDescriptors, "../images/test3DSC.png", true, 50);
}

void FindMeshSpatialSpan(ShapeDescriptor::cpu::Mesh &mesh)
{
    ShapeDescriptor::cpu::float3 meshCenter = {0.0, 0.0, 0.0};

    for(unsigned int i = 0; i < mesh.vertexCount; i++)
    {
        meshCenter += mesh.vertices[i];   
    }

    meshCenter.x /= float(mesh.vertexCount);
    meshCenter.y /= float(mesh.vertexCount);
    meshCenter.z /= float(mesh.vertexCount);

    float maxDistance = 0.0f;
    float maxOriginDistance = 0.0f;
    ShapeDescriptor::cpu::float3 maxDistancePosition = {0.0, 0.0, 0.0};

    std::cout << "Center: " << meshCenter.x << ", " << meshCenter.y << ", " << meshCenter.z << std::endl;

    for(unsigned int i = 0; i < mesh.vertexCount; i++)
    {
        float originDistance = length(mesh.vertices[i]);
        if(originDistance > maxOriginDistance)
        {
            maxOriginDistance = originDistance;
        }
        auto vertexDistanceFromMeshCenter = mesh.vertices[i] - meshCenter;
        float distance = length(vertexDistanceFromMeshCenter);
        if( distance > maxDistance)
        {
            maxDistance = distance;
            maxDistancePosition = mesh.vertices[i];
        }  
    }

    std::cout << "max: " << maxDistance << std::endl;
    std::cout << "origin max: " << maxOriginDistance << std::endl;
    // std::cout << "Center: " << meshCenter.x << ", " << meshCenter.y << ", " << meshCenter.z << std::endl;

}

void PlaceTargetRightOfReference(Model &target, Model &reference){

    auto newPosition = reference.GetPosition();

    float referenceMaxX = 0.5f * reference.GetBoundSpan().x + reference.GetBoundCenter().x;

    newPosition.x += referenceMaxX + 0.5f * target.GetBoundSpan().x - target.GetBoundCenter().x;

    target.SetPosition(newPosition);
}

void PlaceTargetLeftOfReference(Model &target, Model &reference){

    auto newPosition = reference.GetPosition();

    float referenceMinX = -0.5f * reference.GetBoundSpan().x + reference.GetBoundCenter().x;

    newPosition.x += referenceMinX - 0.5f * target.GetBoundSpan().x - target.GetBoundCenter().x;

    target.SetPosition(newPosition);
}

inline bool RangesOverlap(float start0, float end0, float start1, float end1)
{
    return !(end0 < start1 || end1 < start0);
}

inline bool RectangleOverlap(glm::vec2 &start0, glm::vec2 &end0, glm::vec2 &start1, glm::vec2 &end1)
{
    return RangesOverlap(start0.x, end0.x, start1.x, end1.x) && RangesOverlap(start0.y, end0.y, start1.y, end1.y);
}

namespace GeometricTerms{
    enum Axis{X = 0, Y = 1, Z = 2};
};

inline glm::vec3 GetModelMin(Model &model)
{
    return model.GetPosition() + model.GetBoundCenter() - 0.5f * model.GetBoundSpan();
}

inline glm::vec3 GetModelMax(Model &model)
{
    return model.GetPosition() + model.GetBoundCenter() + 0.5f * model.GetBoundSpan();
}

inline glm::vec2 GetVectorWithoutComponent(const glm::vec3 &vector, GeometricTerms::Axis component)
{
    switch (component)
    {
    case GeometricTerms::Axis::X:
        return glm::vec2(vector.y, vector.z);
        break;
    case GeometricTerms::Axis::Y:
        return glm::vec2(vector.x, vector.z);
        break;
    case GeometricTerms::Axis::Z:
        return glm::vec2(vector.x, vector.y);
        break;
    
    default:
        throw std::out_of_range ("Invalid Axis");
        break;
    }
}

inline float GetVectorComponent(const glm::vec3 &vec3, const GeometricTerms::Axis component)
{
    switch (component)
    {
    case BoundingBoxUtilities::GeometricTerms::Axis::X :
        return vec3.x;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Y :
        return vec3.y;
        break;

    case BoundingBoxUtilities::GeometricTerms::Axis::Z :
        return vec3.z;
        break;
    
    default:
        throw std::out_of_range ("float3 index out of range");
        break;
    }
}

bool ModelBoundsTouchAlongAxisUnderMovement(Model &target, Model &reference, glm::vec3 movement, GeometricTerms::Axis axis)
{
    glm::vec2 referenceMin = GetVectorWithoutComponent(GetModelMin(reference), axis);
    glm::vec2 referenceMax = GetVectorWithoutComponent(GetModelMax(reference), axis);

    glm::vec2 targetMin = GetVectorWithoutComponent(GetModelMin(target) + movement, axis);
    glm::vec2 targetMax = GetVectorWithoutComponent(GetModelMax(target) + movement, axis);

    return RectangleOverlap(referenceMin, referenceMax, targetMin, targetMax);
}

float FindMaxDistanceInDirectionWhereModelsBoundsTouch(Model &target, Model &reference, glm::vec3 direction)
{
    const auto targetMinReferenceMaxDistances = (GetModelMax(reference) - GetModelMin(target)) / direction;
    const auto targetMaxReferenceMinDistances = (GetModelMin(reference) - GetModelMax(target)) / direction;

    const std::vector<glm::vec3> oppositeSidesDistances = {targetMinReferenceMaxDistances, targetMaxReferenceMinDistances};
    const std::vector<GeometricTerms::Axis> axes = {GeometricTerms::Axis::X, GeometricTerms::Axis::Y, GeometricTerms::Axis::Z};

    float maxDistance = 0.0f;

    for(auto distances : oppositeSidesDistances)
        for(auto axis : axes)
        {
            const float distance = GetVectorComponent(distances, axis);
            if(distance > maxDistance && ModelBoundsTouchAlongAxisUnderMovement(target, reference, distance * direction, axis))
                maxDistance = distance;
        }

    return maxDistance;
}

glm::vec3 Create3DVectorPointingRandomly()
{

    float azimuth = glm::radians(360.0f * (rand() / static_cast <float> (RAND_MAX)));
    float altitude = glm::radians(180.0f * (rand() / static_cast <float> (RAND_MAX)) - 90.0f);

    return glm::vec3(
        glm::rotate(glm::mat4(1.0f), altitude, glm::vec3(1.0f, 0.0f, 0.0f))
      * glm::rotate(glm::mat4(1.0f), azimuth, glm::vec3(0.0f, 1.0f, 0.0f))
      * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
    );
}

static bool MOVEUNTILLINTERSECTIONUSINGCGAL = false;

void CreateClutteredScene(Model* modelPointer, std::vector<Model*> clutterModelPointers){

    std::vector<Model*> sceneModelPointers;
    sceneModelPointers.push_back(modelPointer);

    for(auto clutterModelPointer : clutterModelPointers)
    {
        auto direction = Create3DVectorPointingRandomly();
        float maxDistance = 0.0f;

        Model* touchingSceneModelPointer = NULL;

        for(auto sceneModelPointer : sceneModelPointers)
        {
            float distance = FindMaxDistanceInDirectionWhereModelsBoundsTouch(*clutterModelPointer, *sceneModelPointer, direction);
            if(distance > maxDistance)
            {
                maxDistance = distance;
                touchingSceneModelPointer = sceneModelPointer;
            }
        }

        clutterModelPointer->SetPosition(direction * maxDistance);
        sceneModelPointers.push_back(clutterModelPointer);
        if(touchingSceneModelPointer != NULL && MOVEUNTILLINTERSECTIONUSINGCGAL)
        {
            moveTargetUntilIntersection(*clutterModelPointer, *touchingSceneModelPointer);
        }
    }

}

BoundingBoxUtilities::BoundingBoxTree CreateTreeFromMesh(ShapeDescriptor::cpu::Mesh &mesh)
{
    ShapeDescriptor::cpu::float3 meshCopy[mesh.vertexCount];
    for (int i = 0; i < mesh.vertexCount; i++){
        meshCopy[i] = mesh.vertices[i];
    }

    auto tree = BoundingBoxUtilities::BoundingBoxTree(meshCopy, mesh.vertexCount, 6);
    return tree;
}

typedef std::vector<BoundingBoxUtilities::BoundingBoxTree*> BoundingBoxForest;


float FindMaxDistanceBetweenTreeAndForest(BoundingBoxUtilities::BoundingBoxTree* tree, BoundingBoxForest forest, glm::vec3 direction)
{


    ShapeDescriptor::cpu::float3 floatDirection = {direction.x, direction.y, direction.z};

    float maxDistance = 0.0f;

    for(auto forestTree : forest)
    {
        float distance = FindMaxValidDistanceUntilBoundsTouch(*tree, *forestTree, floatDirection, maxDistance);

        if(distance > maxDistance)
            maxDistance = distance;
    }


    return maxDistance;

}

typedef std::vector<Model*> ModelPointers;

// #define TimeClutteredSceneCreator 0

void CreateClutteredSceneWithModels(ModelPointers modelPointers)
{
    #ifdef TimeClutteredSceneCreator
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    BoundingBoxForest forest;
    for(auto modelPointer : modelPointers)
    {
        auto direction = Create3DVectorPointingRandomly();
        float distance = FindMaxDistanceBetweenTreeAndForest(modelPointer->GetTreePointer(), forest, direction);
        auto newPosition = modelPointer->GetPosition() + distance * direction;
        modelPointer->SetPosition(newPosition);
        forest.push_back(modelPointer->GetTreePointer());
    }   

    #ifdef TimeClutteredSceneCreator
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "microseconds: " << duration.count() << std::endl;
    #endif
}

void PairIndicesOfEqualPoints(
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> larger,
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> smaller,
    std::vector<IndexPair>* pairs
    )
{

    std::unordered_map<ShapeDescriptor::OrientedPoint, size_t> indexMapping;

    for(size_t i = 0; i < larger.length; i++)
    {
        ShapeDescriptor::OrientedPoint &currentPoint = larger[i];
        indexMapping.insert({currentPoint, i});
    }

    pairs->reserve(smaller.length);
    pairs->clear();

    for(size_t i = 0; i < smaller.length; i++)
    {
        ShapeDescriptor::OrientedPoint &currentPoint = smaller[i];

        if(indexMapping.count(currentPoint) > 0)
        {
            IndexPair pair = {i, indexMapping.at(currentPoint)};
            pairs->emplace_back(pair);
        }
            
    }
    
    pairs->shrink_to_fit();
}

inline ShapeDescriptor::cpu::float3 convertFromVec3(glm::vec3 &vector)
{
    return ShapeDescriptor::cpu::float3(vector.x, vector.y, vector.z);
}

void FillMeshWithModelVertices(ShapeDescriptor::cpu::Mesh &mesh, ModelPointers modelPointers)
{
    size_t meshesVertexCount = 0;
    for(auto pointer : modelPointers)
        meshesVertexCount += pointer->GetMesh().vertexCount;

    ShapeDescriptor::cpu::Mesh outMesh(meshesVertexCount);

    size_t i = 0;

    for(auto pointer : modelPointers)
    {
        glm::vec3 position = pointer->GetPosition();
        position /= pointer->GetScale();
        ShapeDescriptor::cpu::float3 modelPosition = convertFromVec3(position);

        for(size_t j = 0; j < pointer->GetMesh().vertexCount; j++)
        {
            outMesh.vertices[i] = pointer->GetMesh().vertices[j] + modelPosition;
            outMesh.normals[i] = pointer->GetMesh().normals[j];
            i++;
        }
    }

    mesh = outMesh.clone();
    ShapeDescriptor::free::mesh(outMesh);
}

void CreateIndexPairsFromOriginalMesh(std::vector<IndexPair> &pairs, ShapeDescriptor::cpu::Mesh &mesh)
{
    pairs.clear();
    pairs.reserve(mesh.vertexCount);

    for(size_t i = 0; i < mesh.vertexCount; i++)
    {
        IndexPair pair = {i, i};
        pairs.emplace_back(pair);
    }
}

namespace EXAMPLE {
    void ShowSceneWithClutter(OpenGLHandler &openGLHandler)
    {
        const std::string t100SrcPath = "../objects/complete_scans/T100.obj";
        const std::string t12SrcPath = "../objects/complete_scans/T12.obj";
        const std::string t13SrcPath = "../objects/complete_scans/T13.obj";
        const std::string t31SrcPath = "../objects/complete_scans/T31.obj";
        const std::string t45SrcPath = "../objects/complete_scans/T45.obj";

        auto t100 = ShapeDescriptor::utilities::loadOBJ(t100SrcPath, true);
        auto t12 = ShapeDescriptor::utilities::loadOBJ(t12SrcPath, true);
        auto t13 = ShapeDescriptor::utilities::loadOBJ(t13SrcPath, true);
        auto t31 = ShapeDescriptor::utilities::loadOBJ(t31SrcPath, true);
        auto t45 = ShapeDescriptor::utilities::loadOBJ(t45SrcPath, true);

        auto model1 = Model(t100);
        auto model2 = Model(t12);
        auto model3 = Model(t13);
        auto model4 = Model(t31);
        auto model5 = Model(t45);

        model1.CreateTree(6);
        model2.CreateTree(6);
        model3.CreateTree(6);
        model4.CreateTree(6);
        model5.CreateTree(6);

        ModelPointers pointers;
        pointers.push_back(&model1);
        pointers.push_back(&model2);
        pointers.push_back(&model3);
        pointers.push_back(&model4);
        pointers.push_back(&model5);
        CreateClutteredSceneWithModels(pointers);

        openGLHandler.AddModel(model1);
        openGLHandler.AddModel(model2);
        openGLHandler.AddModel(model3);
        openGLHandler.AddModel(model4);
        openGLHandler.AddModel(model5);

        ShapeDescriptor::free::mesh(t100);
        ShapeDescriptor::free::mesh(t12);
        ShapeDescriptor::free::mesh(t13);
        ShapeDescriptor::free::mesh(t31);
        ShapeDescriptor::free::mesh(t45);
    }

    void CreateSingleMeshFromClutter(ShapeDescriptor::cpu::Mesh &mesh)
    {
        const std::string t100SrcPath = "../objects/complete_scans/T100.obj";
        const std::string t12SrcPath = "../objects/complete_scans/T12.obj";
        const std::string t13SrcPath = "../objects/complete_scans/T13.obj";
        const std::string t31SrcPath = "../objects/complete_scans/T31.obj";
        const std::string t45SrcPath = "../objects/complete_scans/T45.obj";

        auto t100 = ShapeDescriptor::utilities::loadOBJ(t100SrcPath, true);
        auto t12 = ShapeDescriptor::utilities::loadOBJ(t12SrcPath, true);
        auto t13 = ShapeDescriptor::utilities::loadOBJ(t13SrcPath, true);
        auto t31 = ShapeDescriptor::utilities::loadOBJ(t31SrcPath, true);
        auto t45 = ShapeDescriptor::utilities::loadOBJ(t45SrcPath, true);

        auto model1 = Model(t100);
        auto model2 = Model(t12);
        auto model3 = Model(t13);
        auto model4 = Model(t31);
        auto model5 = Model(t45);

        model1.CreateTree(6);
        model2.CreateTree(6);
        model3.CreateTree(6);
        model4.CreateTree(6);
        model5.CreateTree(6);

        ModelPointers pointers;
        pointers.push_back(&model1);
        pointers.push_back(&model2);
        pointers.push_back(&model3);
        pointers.push_back(&model4);
        pointers.push_back(&model5);

        CreateClutteredSceneWithModels(pointers);

        FillMeshWithModelVertices(mesh, pointers);

        ShapeDescriptor::free::mesh(t100);
        ShapeDescriptor::free::mesh(t12);
        ShapeDescriptor::free::mesh(t13);
        ShapeDescriptor::free::mesh(t31);
        ShapeDescriptor::free::mesh(t45);

    }
}

namespace DistanceEvaluation {

    enum class DescriptorType {
        QUICCI,
    };

    enum class DistanceType {
        Cross,
    };

    class DistanceEvaluator {

        public: 
            virtual void ComputeInternalElementWise(std::vector<IndexPair> &pairs) = 0;
            virtual void ComputeCrossWise() = 0;
            // virtual void ComputeExternal() = 0;

            // virtual DistanceEvaluator* GetAsDistanceEvaluator() = 0;

    };

    class QUICCIDistanceEvaluator : public DistanceEvaluator {

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors;
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors;

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptorFromMesh(ShapeDescriptor::cpu::Mesh &mesh)
        {

            auto gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

            auto descriptorOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
            auto gpuDescriptorOrigins = ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);

            std::cout << descriptorOrigins.length << std::endl;

            float supportRadius = 15.0f;

            auto descriptors = ShapeDescriptor::gpu::generateQUICCImages(
                gpuMesh,
                gpuDescriptorOrigins,
                supportRadius);

            ShapeDescriptor::free::mesh(gpuMesh);

            return descriptors;
        }

        public: 

            QUICCIDistanceEvaluator(ShapeDescriptor::cpu::Mesh &needleMesh, ShapeDescriptor::cpu::Mesh &otherMesh)
            {
                needleDescriptors = computeDescriptorFromMesh(needleMesh);
                haystackDescriptors = computeDescriptorFromMesh(otherMesh);
            }

            void ComputeInternalElementWise(std::vector<IndexPair> &pairs)
            {
                ShapeDescriptor::cpu::array<IndexPair> cpuPairs(pairs.size());
                std::copy(pairs.begin(), pairs.end(), cpuPairs.content);

                ShapeDescriptor::gpu::array<IndexPair> gpuPairs = ShapeDescriptor::copy::hostArrayToDevice(cpuPairs);

                DescriptorDistance::Hamming::FindElementWiseDistances(needleDescriptors, haystackDescriptors, gpuPairs);
                
                ShapeDescriptor::free::array(cpuPairs);
                ShapeDescriptor::free::array(gpuPairs);
            }

            void ComputeCrossWise()
            {
                DescriptorDistance::Hamming::FindMinDistance(needleDescriptors, haystackDescriptors);
            }

            // void ComputeExternal() = 0;

            // virtual DistanceEvaluator* GetAsDistanceEvaluator();

    };

    void ComputeDistance(DistanceEvaluator &evaluator, DistanceType type)
    {
        switch (type)
        {
        case DistanceType::Cross :
            evaluator.ComputeCrossWise();
            break;
        }

    }


    void ComputeCrossWise(
        DescriptorType descriptorType,
        DistanceType distanceType,
        ShapeDescriptor::cpu::Mesh &needleMesh,
        ShapeDescriptor::cpu::Mesh &otherMesh
        )
    {
        
        switch (descriptorType)
        {
        case DescriptorType::QUICCI:
            {
            auto derivedEvaluator = QUICCIDistanceEvaluator(needleMesh, otherMesh);
            auto &evaluator = dynamic_cast<DistanceEvaluator&>(derivedEvaluator);

            ComputeDistance(evaluator, distanceType);

            break;
            }
        }

    }

}

class DescriptorTester {

    public:

        enum class VisualizeObject {
            None,
            Needle,
            Altered,
            ClutterWithBoundindBoxes
        };

        enum class SaveDescriptor {
            None,
            Needle,
            Other,
            Both,
        };


        enum class EvaluationType {
            None,
            Internal,
            InternalElementWise,
            External,
        };

    private:

        OpenGLHandler openGLHandler;

        bool applyClutter = false, applyOcclusion = false, applyNoise = false;

        VisualizeObject visualizeObject = VisualizeObject::None;
        EvaluationType evaluationType = EvaluationType::None;
        SaveDescriptor saveDescriptor = SaveDescriptor::None;

        DistanceEvaluation::DescriptorType descriptorType = DistanceEvaluation::DescriptorType::QUICCI; 

        std::string needleMeshPath = "../objects/complete_scans/T100.obj";

        std::vector<std::string> clutterMeshPaths = {
            "../objects/complete_scans/T100.obj",
            "../objects/complete_scans/T12.obj",
            "../objects/complete_scans/T13.obj",
            "../objects/complete_scans/T31.obj",
            "../objects/complete_scans/T45.obj"
        };

        std::vector<std::string> externalMeshPaths = {
            "../objects/complete_scans/T100.obj",
            "../objects/complete_scans/T12.obj",
            "../objects/complete_scans/T13.obj",
            "../objects/complete_scans/T31.obj",
            "../objects/complete_scans/T45.obj"
        };

        std::vector<IndexPair> indexPairs;

        ShapeDescriptor::cpu::Mesh needleMesh = ShapeDescriptor::utilities::loadOBJ(needleMeshPath, true);
        ShapeDescriptor::cpu::Mesh otherMesh = ShapeDescriptor::utilities::loadOBJ(needleMeshPath, true);

        // size_t maxNeedleDescriptorCount = 100000;
        // size_t maxOtherDescriptorCount = 100000;

        // bool reduceDescriptorsRandomly = true;

        float noiseScale = 0.0f;

        void ApplyDisturbances()
        {
            if(evaluationType == EvaluationType::InternalElementWise)
            {
                indexPairs.reserve(needleMesh.vertexCount);
                for(size_t i = 0; i < needleMesh.vertexCount; i++)
                {
                    IndexPair pair = {i, i};
                    indexPairs.emplace_back(pair);
                }
            }


            if(applyClutter)
            {
                std::vector<ShapeDescriptor::cpu::Mesh> meshes;
                meshes.reserve(clutterMeshPaths.size());

                for(const auto &path : clutterMeshPaths)
                    meshes.emplace_back(ShapeDescriptor::utilities::loadOBJ(path, true));

                std::vector<Model> models;
                models.reserve(clutterMeshPaths.size());

                for(auto &mesh : meshes)
                    models.emplace_back(Model(mesh, glm::vec3(0.0f), meshTypes::Scaled));

                unsigned int treeDepth = 6;

                for(auto &model : models)
                    model.CreateTree(treeDepth);

                ModelPointers pointers;
                for(auto &model : models)
                    pointers.push_back(&model);

                CreateClutteredSceneWithModels(pointers);

                if(visualizeObject == VisualizeObject::ClutterWithBoundindBoxes)
                {
                    for(const auto &model : models)
                        openGLHandler.AddModel(model);
                }

                FillMeshWithModelVertices(otherMesh, pointers);

                for(auto &mesh : meshes)
                    ShapeDescriptor::free::mesh(mesh);

            }

            if(applyOcclusion)
            {
                auto &occlusionProvider = openGLHandler.GetOcclusionProvider();

                std::unordered_map<size_t, size_t> mapping;
                
                if(evaluationType == EvaluationType::InternalElementWise)
                    occlusionProvider.SetIndexMapping(&mapping);
                
                occlusionProvider.CreateMeshWithOcclusion(otherMesh);

                if(evaluationType == EvaluationType::InternalElementWise)
                {
                    std::vector<IndexPair> pairs;
                    pairs.reserve(needleMesh.vertexCount);
                    for(const auto &pair : indexPairs)
                    {
                        IndexPair newPair = pair;
                        size_t left =  mapping.at(pair.right);

                        if(left >= needleMesh.vertexCount)
                            break;
                        
                        pairs.emplace_back(newPair);
                    }   

                    pairs.shrink_to_fit();
                    indexPairs = pairs;
                }
            }

            if(applyNoise)
            {
                auto vertexMap = MeshFunctions::MapVertexIndices(&otherMesh);
                MeshFunctions::MoveVerticesAlongAverageNormal(&otherMesh, vertexMap, noiseScale);
            }

            if(visualizeObject == VisualizeObject::Altered)
            {
                auto model = Model(otherMesh, glm::vec3(0.0f), meshTypes::Scaled);
                openGLHandler.AddModel(model);
            }
        }

        void Draw()
        {

            if(visualizeObject == VisualizeObject::None)
                return;
      
            if(visualizeObject == VisualizeObject::Needle)
            {
                auto model = Model(needleMesh, glm::vec3(0.0f), meshTypes::Scaled);
                openGLHandler.AddModel(model);
            }

            openGLHandler.Draw();

        }

        void TestDistances()
        {
                 
            switch (evaluationType)
            {
            case EvaluationType::Internal : 
                ComputeCrossWise( descriptorType, DistanceEvaluation::DistanceType::Cross, needleMesh, otherMesh);
                break;
                
            }

        }

    public:


        OpenGLHandler &GetOpenGLHandlerReference()
        {
            return openGLHandler;
        }

        void SetDrawMode(VisualizeObject mode)
        {
            visualizeObject = mode;
        }

        void SetTestMode(EvaluationType mode)
        {
            evaluationType = mode;
        }

        void SetApplyClutter(bool mode)
        {
            applyClutter = mode;
        }

        void SetApplyOcclusion(bool mode)
        {
            applyOcclusion = mode;
        }

        void SetApplyNoise(bool mode)
        {
            applyNoise = mode;
        }

        void SetNoiseScale(float scale)
        {
            noiseScale = scale;
        }

        void SetOcclusionViewPoint(glm::vec3 newViewPoint)
        {
            OcclusionProvider &occlusionProvider = openGLHandler.GetOcclusionProvider();
            occlusionProvider.SetViewPoint(newViewPoint);
        }

        void Run()
        {
            ApplyDisturbances();

            Draw();

            TestDistances();
        }

};


int main()
{
    DescriptorTester tester;

    tester.SetApplyClutter(true);
    tester.SetApplyOcclusion(true);
    tester.SetApplyNoise(true);

    tester.SetNoiseScale(1.0f);

    tester.SetTestMode(DescriptorTester::EvaluationType::Internal);
    tester.SetDrawMode(DescriptorTester::VisualizeObject::Altered);
    
    tester.Run();

    std::cout << "closing program" << std::endl;
}