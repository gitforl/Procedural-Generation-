#pragma once

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>

class MeshModifier {
    private:
        std::string objSrcPath;
        ShapeDescriptor::cpu::Mesh mesh;
    public:
        MeshModifier(std::string objSrcPath);
        ~MeshModifier();
        void CheckMesh();
};