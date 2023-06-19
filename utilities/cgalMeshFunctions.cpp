#include "cgalMeshFunctions.hpp"

CGALMesh createCGALMesh(ShapeDescriptor::cpu::Mesh &mesh){
    CGALMesh cgalMesh;

    for(unsigned int i = 0; i < mesh.vertexCount / 3; i++)
    {

        auto v0 = mesh.vertices[i * 3 + 0];
        auto v1 = mesh.vertices[i * 3 + 1];
        auto v2 = mesh.vertices[i * 3 + 2];

        // auto gv0 = glm::vec4(v0.x, v0.y, v0.z, 1.0f);
        // auto gv1 = glm::vec4(v1.x, v1.y, v1.z, 1.0f);
        // auto gv2 = glm::vec4(v2.x, v2.y, v2.z, 1.0f);

        // auto m0 = transformation * gv0;
        // auto m1 = transformation * gv1;
        // auto m2 = transformation * gv2;

        // auto p0 = Point3(v0.x + translation.x, v0.y + translation.y, v0.z + translation.z);
        // auto p1 = Point3(v1.x + translation.x, v1.y + translation.y, v1.z + translation.z);
        // auto p2 = Point3(v2.x + translation.x, v2.y + translation.y, v2.z + translation.z);

        auto p0 = Point3(v0.x, v0.y, v0.z);
        auto p1 = Point3(v1.x, v1.y, v1.z);
        auto p2 = Point3(v2.x, v2.y, v2.z);

        // auto p0 = Point3(m0.x, m0.y, m0.z);
        // auto p1 = Point3(m1.x, m1.y, m1.z);
        // auto p2 = Point3(m2.x, m2.y, m2.z);

        CGALMesh::vertex_index u0 = cgalMesh.add_vertex(p0);
        CGALMesh::vertex_index u1 = cgalMesh.add_vertex(p1);
        CGALMesh::vertex_index u2 = cgalMesh.add_vertex(p2);

        cgalMesh.add_face(u0, u1, u2);
    }

    return cgalMesh;
}

bool meshesIntersect(ShapeDescriptor::cpu::Mesh &mesh1, ShapeDescriptor::cpu::Mesh &mesh2){

    const float distanceThreshold = 0.01f;
    float distance = 0.0f;

    auto t = CGAL::Timer();

    
    CGALMesh cgalMesh1 = createCGALMesh(mesh1);
    CGALMesh cgalMesh2 = createCGALMesh(mesh2);

    t.start();

    float directionScaler = 0.02f;
    Vector_3 transvec(0.0, -directionScaler, 0.0);
    Aff_transformation_3 transl(CGAL::TRANSLATION, transvec);

    Aff_transformation_3 scaling(CGAL::SCALING, 0.01f);
    CGAL::Polygon_mesh_processing::transform(scaling, cgalMesh1);
    CGAL::Polygon_mesh_processing::transform(scaling, cgalMesh2);

    const Vector_3 baseTransvec(0.0, 1.5, 0.0);
    Aff_transformation_3 baseTransl(CGAL::TRANSLATION, baseTransvec);
    CGAL::Polygon_mesh_processing::transform(baseTransl, cgalMesh1);


    auto result = CGAL::Polygon_mesh_processing::do_intersect(cgalMesh1, cgalMesh2);
    if(result)
        std::cout << "meshes already intersect" << std::endl;
    

    for(int i = 0; i < 15; i++){
        CGAL::Polygon_mesh_processing::transform(transl, cgalMesh1);
        result = CGAL::Polygon_mesh_processing::do_intersect(cgalMesh1, cgalMesh2);
        if(result)
            break;

        distance += directionScaler;
    }

    if(result)
    transvec = Vector_3(0.0, directionScaler, 0.0);
    transl = Aff_transformation_3(CGAL::TRANSLATION, transvec);
    CGAL::Polygon_mesh_processing::transform(transl, cgalMesh1);

    std::cout << "distance: " << distance << std::endl;

    t.stop();
    std::cout << "time: " << t.time() << std::endl;

    return true;
}

static bool SIMPLIFYMESH = true;

void moveTargetUntilIntersection(Model &target, Model &reference)
{
    glm::vec3 direction = 0.01f * glm::normalize(reference.GetPosition() - target.GetPosition());
    moveTargetUntilIntersectionAlongDirection(target, reference, direction);
}

void moveTargetUntilIntersectionAlongDirection(Model &target, Model &reference, glm::vec3 direction)
{
    CGALMesh cgalTarget = createCGALMesh(target.GetMesh());
    CGALMesh cgalReference = createCGALMesh(reference.GetMesh());

    Aff_transformation_3 targetScaling(CGAL::SCALING, target.GetScale());
    CGAL::Polygon_mesh_processing::transform(targetScaling, cgalTarget);

    Aff_transformation_3 referenceScaling(CGAL::SCALING, reference.GetScale());
    CGAL::Polygon_mesh_processing::transform(referenceScaling, cgalReference);

    Vector_3 targetTransvec(target.GetPosition().x, target.GetPosition().y, target.GetPosition().z);
    Aff_transformation_3 targetTranslation(CGAL::TRANSLATION, targetTransvec);

    Vector_3 referenceTransvec(reference.GetPosition().x, reference.GetPosition().y, reference.GetPosition().z);
    Aff_transformation_3 referenceTranslation(CGAL::TRANSLATION, referenceTransvec);

    CGAL::Polygon_mesh_processing::transform(targetTranslation, cgalTarget);
    CGAL::Polygon_mesh_processing::transform(referenceTranslation, cgalReference);

    float distance = distanceBeforeIntersection(cgalTarget, cgalReference, direction);

    std::cout << "distance: " << distance << std::endl;

    glm::vec3 newPosition = target.GetPosition() + distance * glm::normalize(direction);
    target.SetPosition(newPosition);
}

unsigned int MAXMOVEMENTITERATIONS = 100;

float distanceBeforeIntersection(CGALMesh &mesh1, CGALMesh &mesh2, glm::vec3 direction){

    float distance = 0.0f;
    float length = glm::length(direction);

    const Vector_3 transvec(direction.x, direction.y, direction.z);
    const Aff_transformation_3 transl(CGAL::TRANSLATION, transvec);

    auto result = CGAL::Polygon_mesh_processing::do_intersect(mesh1, mesh2);
    if(result)
        std::cout << "meshes already intersect" << std::endl;
    
    for(int i = 0; i < MAXMOVEMENTITERATIONS; i++)
    {
        CGAL::Polygon_mesh_processing::transform(transl, mesh1);
        result = CGAL::Polygon_mesh_processing::do_intersect(mesh1, mesh2);
        if(result)
            break;

        distance += length;
    }

    return distance;
}