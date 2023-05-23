#pragma once

#include <openglHandler/openglMesh.hpp>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <utilities/boundingBox.hpp>


class Model {
    private:
        ShapeDescriptor::cpu::Mesh mesh;
        OpenGLMesh openGlmesh;
        MeshFunctions::boundingBox bound;
        float scale = 0.01f;
        glm::vec3 position;

        BoundingBoxUtilities::BoundingBoxTree *treePointer = NULL;

    public:
        Model(ShapeDescriptor::cpu::Mesh &mesh, glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), meshTypes meshType = meshTypes::Occlusion);
        ShapeDescriptor::cpu::Mesh& GetMesh();
        BoundingBoxUtilities::BoundingBoxTree* GetTreePointer();
        glm::vec3 GetBoundCenter();
        glm::vec3 GetBoundSpan();
        glm::vec3 GetPosition();
        void CreateTree(unsigned int depth);
        float GetScale();
        void SetPosition(glm::vec3 newPosition);
        void Draw();

};