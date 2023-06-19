#include "model.hpp"


inline ShapeDescriptor::cpu::float3 convertVec3ToFloat3(glm::vec3 vector)
{
    return {vector.x, vector.y, vector.z};
}

Model::Model(ShapeDescriptor::cpu::Mesh &mesh, glm::vec3 position, meshTypes meshType):
mesh(mesh),
openGlmesh(OpenGLMesh(mesh, meshType)),
bound(MeshFunctions::boundingBox(mesh)),
position(position)
{

}
ShapeDescriptor::cpu::Mesh & Model::GetMesh()
{
    return mesh;
}

BoundingBoxUtilities::BoundingBoxTree* Model::GetTreePointer()
{
    return treePointer;
}

glm::vec3 Model::GetBoundCenter(){

    glm::vec3 center = glm::vec3
    (
        scale * bound.center().x, 
        scale * bound.center().y, 
        scale * bound.center().z
    );

    return center;
}

glm::vec3 Model::GetBoundSpan(){

    glm::vec3 span = glm::vec3
    (
        scale * bound.span().x, 
        scale * bound.span().y, 
        scale * bound.span().z
    );

    return span;
}

glm::vec3 Model::GetPosition()
{
    return position;
}

float Model::GetScale()
{
    return scale;
}

void Model::SetPosition(glm::vec3 newPosition){
    position = newPosition;
    treePointer->setTranslation(convertVec3ToFloat3(position));
}

void Model::Draw(){
    openGlmesh.Draw();
}

inline void CopyVerticesFromMesh(ShapeDescriptor::cpu::float3 *copy, ShapeDescriptor::cpu::Mesh &mesh)
{
    for (int i = 0; i < mesh.vertexCount; i++){
        copy[i] = mesh.vertices[i];
    }
}

void Model::CreateTree(unsigned int depth)
{
    if(treePointer == NULL)
    {
        ShapeDescriptor::cpu::float3 verticesCopy[mesh.vertexCount];
        CopyVerticesFromMesh(verticesCopy, mesh);

        treePointer = new BoundingBoxUtilities::BoundingBoxTree(verticesCopy, mesh.vertexCount, depth);

        treePointer->setScale(scale);
        treePointer->setTranslation(convertVec3ToFloat3(position));
    }
}