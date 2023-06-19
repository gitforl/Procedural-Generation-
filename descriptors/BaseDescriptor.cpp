#include "BaseDescriptor.hpp"

BaseDescriptor::BaseDescriptor(std::string objSrcPath):
objSrcPath(objSrcPath) {
    InitializeMesh();
    InitializeAlteredMesh();
    InitializeDescriptorOrigins();

    vertexMap = MeshFunctions::MapVertexIndices(&mesh);
}
BaseDescriptor::BaseDescriptor()
{
    std::cout << "partial init of base descriptor" << std::endl;
}

void BaseDescriptor::InitializeMesh() {
    mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
}

void BaseDescriptor::InitializeAlteredMesh() {
    alteredMesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    alteredGpuMesh = ShapeDescriptor::copy::hostMeshToDevice(alteredMesh);
}

void BaseDescriptor::InitializeDescriptorOrigins() {
    descriptorOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(mesh);
    gpuDescriptorOrigins = ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);
}

void BaseDescriptor::ApplyNoise(float noiseLevel){
    alteredMesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    MeshFunctions::MoveVerticesAlongAverageNormal(&alteredMesh, vertexMap, noiseLevel);
}

void BaseDescriptor::ComputeAverageDistance(){
    averageDistance = computeFloatAverage(comparisonValues);
}

void BaseDescriptor::ComputeStandardDeviation(){
    standardDeviation = computeFloatStandardDeviation(comparisonValues, averageDistance);
}

void BaseDescriptor::RunNoiseTestAtLevel(float noiseLevel){

    ApplyNoise(noiseLevel);
    CreateReferenceDescriptors();
    CreateAlteredDescriptors();
    Compare();
    ComputeAverageDistance();
    ComputeStandardDeviation();

    std::cout << "average distance: " << averageDistance << std::endl;
    std::cout << "standard deviation: " << standardDeviation << std::endl;
}

void BaseDescriptor::RunNoiseTestAtVaryingLevels(std::vector<float> noiseLevels){

    CreateReferenceDescriptors();

    for(float noiseLevel : noiseLevels){
        ApplyNoise(noiseLevel);
        CreateAlteredDescriptors();
        Compare();
        ComputeAverageDistance();
        ComputeStandardDeviation();

        std::cout << "average distance: " << averageDistance << std::endl;
        std::cout << "standard deviation: " << standardDeviation << std::endl;
    }
}



void BaseDescriptor::MeshSelfIntersects(){
    CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> CGAL_mesh;

    if(!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(objSrcPath, CGAL_mesh) || !CGAL::is_triangle_mesh(CGAL_mesh)){
        std::cerr << "Invalid input." << std::endl;
        return;
    }

    CGAL::Real_timer timer;
    timer.start();

    bool intersecting = CGAL::Polygon_mesh_processing::does_self_intersect<CGAL::Parallel_if_available_tag>(CGAL_mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, CGAL_mesh)));
    std::cout << (intersecting ? "Yes" : "No") << std::endl;
    std::cout << "Elapsed time: " << timer.time() << std::endl;

}


void BaseDescriptor::CGALMeshTest(){
    CGAL::Real_timer timer;
    timer.start();
    CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> bowl_mesh;

    if(!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(objSrcPath, bowl_mesh) || !CGAL::is_triangle_mesh(bowl_mesh)){
        std::cerr << "Invalid input." << std::endl;
        return;
    }

    CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> boat_mesh;

    if(!CGAL::Polygon_mesh_processing::IO::read_polygon_mesh(objSrcPath, boat_mesh) || !CGAL::is_triangle_mesh(boat_mesh)){
        std::cerr << "Invalid input." << std::endl;
        return;
    }

    std::cout << "time: " << timer.time() << std::endl;

    std::cout << "num faces: " << bowl_mesh.num_faces() << std::endl;
    std::cout << "joined: " << bowl_mesh.join(boat_mesh) << std::endl;
    std::cout << "num faces: " << bowl_mesh.num_faces() << std::endl;






}