#pragma once

#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/transform.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/aff_transformation_tags.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Vector_3.h>

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>

#include <CGAL/Timer.h>

#include <meshModifier/model.hpp>
#include <shapeDescriptor/utilities/free/mesh.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point3;
typedef CGAL::Surface_mesh<Point3> CGALMesh;
typedef Kernel::Vector_3 Vector_3;
typedef CGAL::Aff_transformation_3<Kernel> Aff_transformation_3;

CGALMesh createCGALMesh(ShapeDescriptor::cpu::Mesh &mesh);
bool meshesIntersect(ShapeDescriptor::cpu::Mesh &mesh1, ShapeDescriptor::cpu::Mesh &mesh2);
float distanceBeforeIntersection(CGALMesh &mesh1, CGALMesh &mesh2, glm::vec3 direction);
void moveTargetUntilIntersection(Model &target, Model &reference);
void moveTargetUntilIntersectionAlongDirection(Model &target, Model &reference, glm::vec3 direction);