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


// bool ModelBoundsTouchAlongAxisUnderMovement2(Model &target, Model &reference, glm::vec3 movement, GeometricTerms::Axis axis)
// {
//     glm::vec3 referenceMin = GetModelMin(reference);
//     glm::vec3 referenceMax = GetModelMax(reference);

//     glm::vec3 targetMin = GetModelMin(target) + movement;
//     glm::vec3 targetMax = GetModelMax(target) + movement;

//     return (RangesOverlap(referenceMin.x, referenceMax.x, targetMin.x, targetMax.x) || axis)
//         && (RangesOverlap(referenceMin.y, referenceMax.y, targetMin.y, targetMax.y) || axis)
//         && (RangesOverlap(referenceMin.z, referenceMax.z, targetMin.z, targetMax.z) || axis);
// }

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

void CreateClutteredSceneWithModels(ModelPointers modelPointers)
{
    auto start = std::chrono::high_resolution_clock::now();

    BoundingBoxForest forest;
    for(auto modelPointer : modelPointers)
    {
        auto direction = Create3DVectorPointingRandomly();
        float distance = FindMaxDistanceBetweenTreeAndForest(modelPointer->GetTreePointer(), forest, direction);
        auto newPosition = modelPointer->GetPosition() + distance * direction;
        modelPointer->SetPosition(newPosition);
        forest.push_back(modelPointer->GetTreePointer());
    }   

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "microseconds: " << duration.count() << std::endl;
}


int main()
{
    OpenGLHandler openGLHandler;

    const std::string t100SrcPath = "../objects/complete_scans/T100.obj";
    const std::string t12SrcPath = "../objects/complete_scans/T12.obj";
    const std::string t13SrcPath = "../objects/complete_scans/T13.obj";
    const std::string t31SrcPath = "../objects/complete_scans/T31.obj";
    const std::string t45SrcPath = "../objects/complete_scans/T45.obj";
    // // const std::string objSrcPath = "../objects/OCCLUDED.obj";
    
    // QUICCIDescriptor testDescriptor(objSrcPath);

    // // testDescriptor.MeshSelfIntersects();
    // testDescriptor.CGALMeshTest();

    // MeshModifier meshModifier(objSrcPath);
    // meshModifier.DrawScreen();

    std::cout << "her" << std::endl;

    auto t100 = ShapeDescriptor::utilities::loadOBJ(t100SrcPath, true);
    auto t12 = ShapeDescriptor::utilities::loadOBJ(t12SrcPath, true);
    auto t13 = ShapeDescriptor::utilities::loadOBJ(t13SrcPath, true);
    auto t31 = ShapeDescriptor::utilities::loadOBJ(t31SrcPath, true);
    auto t45 = ShapeDescriptor::utilities::loadOBJ(t45SrcPath, true);

    // auto oglMesh = OpenGLMesh(mesh, meshTypes::Occlusion);
    // oglMesh.SetPosition(glm::vec3(0.0f, 2.0f, 0.0f));


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

    std::cout << "her" << std::endl;



    // auto direction = glm::vec3(-1.0f, 0.0f, 0.5f);  

    // auto dist = FindMaxDistanceInDirectionWhereModelsBoundsTouch(model2, model1, direction);

    // model2.SetPosition(direction * dist);


    // PlaceTargetLeftOfReference(model2, model1);
    // PlaceTargetRightOfReference(model2, model1);

    // moveTargetUntilIntersection(model2, model1);

    // model2.SetPosition(glm::vec3(0.0f, 2.0f, 0.0f));

    openGLHandler.AddModel(model1);
    openGLHandler.AddBoundTree(*model1.GetTreePointer());
    openGLHandler.AddModel(model2);
    openGLHandler.AddBoundTree(*model2.GetTreePointer());
    openGLHandler.AddModel(model3);
    openGLHandler.AddBoundTree(*model3.GetTreePointer());
    openGLHandler.AddModel(model4);
    openGLHandler.AddBoundTree(*model4.GetTreePointer());
    openGLHandler.AddModel(model5);
    openGLHandler.AddBoundTree(*model5.GetTreePointer());

    // std::cout << meshesIntersect(mesh, t12) << std::endl;

    openGLHandler.Draw();
    // openGLHandler.CreateMeshFromVisibleTriangles();

    // FindMeshSpatialSpan(mesh);

    // meshModifier.CheckMesh();

    // std::vector<float> noiseLevels({0.0f, 0.01f, 0.1f, 1.0f});

    // testDescriptor.RunNoiseTestAtLevel(0.0f);
    // testDescriptor.RankDescriptors();
    // testDescriptor.RunNoiseTestAtLevel(0.1f);

    // testDescriptor.RankDescriptors();


    // testDescriptor.RunNoiseTestAtVaryingLevels(noiseLevels);

    std::cout << "closing program" << std::endl;

}