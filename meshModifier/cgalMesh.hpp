#pragma once

#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/transform.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/aff_transformation_tags.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Vector_3.h>

#include <CGAL/Timer.h>

#include <glm/glm.hpp>
#include <shapeDescriptor/utilities/free/mesh.h>

// typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
// typedef Kernel::Point_3 Point3;
// typedef CGAL::Surface_mesh<Point3> CGALMesh;
// typedef Kernel::Vector_3 Vector_3;
// typedef CGAL::Aff_transformation_3<Kernel> Aff_transformation_3;

// class CGALMesh {
//     private:
//         CGALMesh cgalMesh;
//         glm::vec3 translation;
//         float scale;

//         void createCGALMesh(ShapeDescriptor::cpu::Mesh &mesh);
//     public:
//         CGALMesh(ShapeDescriptor::cpu::Mesh &mesh, glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f), float scale = 1.0f);
// };