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

#include <utilities/meshFunctions.hpp>
#include <utilities/generalUtilities.hpp>

#include <meshModifier/model.hpp>
#include <utilities/boundingBox.hpp>

class OcclusionProvider {

    protected:

        std::unordered_map<size_t, size_t> *mapping = nullptr;
        glm::vec3 viewPoint = glm::vec3(0.0f, 0.0f, 10.0f);

    public:

        void SetViewPoint(glm::vec3 newViewPoint)
        {
            viewPoint = newViewPoint;
        }

        void SetIndexMapping(std::unordered_map<size_t, size_t> *newMapping)
        {
            mapping = newMapping;
        }

        virtual void CreateMeshWithOcclusion(ShapeDescriptor::cpu::Mesh &mesh) = 0; 

};

class OpenGLHandler : OcclusionProvider {
    private:
        const unsigned int SCR_WIDTH = 800;
        const unsigned int SCR_HEIGHT = 600;
        glm::mat4 projection;
        GLFWwindow* window;
        Shader shader;

        std::vector<OpenGLMesh> meshes;
        // std::vector<MeshFunctions::boundingBox> boundingBoxes;
        std::vector<Model> models;

        BoundingBoxUtilities::BoundingBoxTree * currentBoundTreePointer = NULL;
        
        //Occlusion

        glm::vec3 occlusionDetectionCameraPosition = glm::vec3(0.0f, 0.0f, 10.0f);

        unsigned int offscreenTextureWidth;
        unsigned int offscreenTextureHeight;

        void DrawLeafBoundingBoxes(BoundingBoxUtilities::BoundingBoxNode * node, glm::mat4 &VP, int lineTransformationLoc, OpenGLMesh &boundMesh);

        unsigned int CreateAndBindFrameBuffer();
        unsigned int CreateAndBindFrameBufferTexture();
        unsigned int CreateAndBindFrameBufferRenderBuffer();
        void PrepareOcclusionDetectionRenderBuffer(unsigned int fbo);
        void SetupOcclusionDetectionShader();
        void CopyTextureToLocalBuffer(unsigned int texture, std::vector<unsigned char> &localFramebufferCopy);
        void CheckIfTriangleAppearsInImage(std::vector<unsigned char> &localFramebufferCopy, std::vector<bool> &triangleAppearsInImage);
        void ConstructMeshFromVisibleTriangles(ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::cpu::Mesh &outMesh, std::vector<bool> &triangleAppearsInImage);
        void CleanUpOcclusionDetection(unsigned int &fbo, unsigned int &texture, unsigned int &rbo);
        
    public:
        OpenGLHandler();
        ~OpenGLHandler();
        void AddMesh(OpenGLMesh mesh);
        void AddModel(Model model);
        void AddBoundingBox(ShapeDescriptor::cpu::Mesh mesh);
        void CreateMeshWithOcclusion(ShapeDescriptor::cpu::Mesh &mesh);//, std::unordered_map<size_t, size_t> *mapping = nullptr);
        void Draw();

        OcclusionProvider &GetOcclusionProvider();
};