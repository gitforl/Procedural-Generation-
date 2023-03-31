#pragma once

#include <iostream>
#include <vector>


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>

#include <openglHandler/openglMesh.hpp>
#include <openglHandler/shader.hpp>

class OpenGLHandler {
    private:
        const unsigned int SCR_WIDTH = 800;
        const unsigned int SCR_HEIGHT = 600;
        glm::mat4 projection;
        GLFWwindow* window;
        Shader shader;

        std::vector<OpenGLMesh> meshes;

        //Occlusion

        glm::vec3 occlusionDetectionCameraPosition = glm::vec3(0.0f, 0.0f, 10.0f);

        unsigned int offscreenTextureWidth;
        unsigned int offscreenTextureHeight;

        unsigned int CreateAndBindFrameBuffer();
        unsigned int CreateAndBindFrameBufferTexture();
        unsigned int CreateAndBindFrameBufferRenderBuffer();
        void PrepareOcclusionDetectionRenderBuffer(unsigned int fbo);
        void SetupOcclusionDetectionShader();
        void CopyTextureToLocalBuffer(unsigned int texture, std::vector<unsigned char> &localFramebufferCopy);
        void CheckIfTriangleAppearsInImage(std::vector<unsigned char> &localFramebufferCopy, std::vector<bool> &triangleAppearsInImage);
        void ConstructMeshFromVisibleTriangles(ShapeDescriptor::cpu::Mesh mesh, ShapeDescriptor::cpu::Mesh outMesh, std::vector<bool> &triangleAppearsInImage);
        
    public:
        OpenGLHandler();
        ~OpenGLHandler();
        void AddMesh(OpenGLMesh mesh);
        void CreateMeshFromVisibleTriangles();
        void Draw();

};