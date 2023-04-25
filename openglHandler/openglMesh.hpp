#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include <shapeDescriptor/utilities/free/mesh.h>
#include <utilities/meshFunctions.hpp>

enum meshTypes { Base, Occlusion, Scaled, Lines};

class OpenGLMesh {
    private:
        unsigned int VBO, VAO;
        unsigned int vertexCount;
        // glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
        meshTypes type = meshTypes::Base;

        void simpleMeshInit(ShapeDescriptor::cpu::Mesh &mesh, float scaler = 1.0f);
        void occlusionMeshInit(ShapeDescriptor::cpu::Mesh &mesh);
        void vertexMeshInit(float* vertices, unsigned int size);
    public:
        OpenGLMesh(ShapeDescriptor::cpu::Mesh &mesh, meshTypes type = meshTypes::Base);
        OpenGLMesh(float* vertices, unsigned int size);
        // void SetPosition(glm::vec3 newPosition);
        // glm::vec3 GetPosition();
        meshTypes GetType();
        void Draw();
        // void UniformModel(unsigned int uniformLocation);
};