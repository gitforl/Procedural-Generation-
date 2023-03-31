#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <shapeDescriptor/utilities/free/mesh.h>

class OpenGLMesh {
    private:
        unsigned int VBO, VAO;
        void simpleMeshInit(ShapeDescriptor::cpu::Mesh &mesh);
        void occlusionMeshInit(ShapeDescriptor::cpu::Mesh &mesh);
        unsigned int vertexCount;
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
    public:
        OpenGLMesh(ShapeDescriptor::cpu::Mesh &mesh);
        void SetPosition(glm::vec3 newPosition);
        void Draw();
        void UniformModel(unsigned int uniformLocation);
};