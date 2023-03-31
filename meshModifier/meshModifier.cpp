#include "meshModifier.hpp"

MeshModifier::MeshModifier(std::string objSrcPath):
objSrcPath(objSrcPath)
{
    mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);
    std::cout << "Mesh Modifier created" << std::endl;
}

MeshModifier::~MeshModifier(){
    std::cout << "Mesh Modifier destroyed" << std::endl;
}

void MeshModifier::CheckMesh(){
    std::cout << "number of vertices: " << mesh.vertexCount << std::endl;

    // indices = array[]

    // for(int i = 0; i < mesh.vertexCount; i += 3)
    // {

    // }
}
