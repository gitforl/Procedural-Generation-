#include "openglHandler.hpp"

glm::mat4 model = glm::mat4(1.0f);
float previousTime = glfwGetTime();

glm::mat4 cameraRotation = glm::mat4(1.0f);
float distanceFromCenter = 5.0f;
glm::mat4 view = glm::lookAt(glm::vec3(.0f, .0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
};
void processInput(GLFWwindow *window)
{

    float currentTime = glfwGetTime();
    float deltaTime = currentTime - previousTime;
    previousTime = currentTime;

    float speedScaler = 0.1f * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraRotation = glm::rotate(cameraRotation, glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraRotation = glm::rotate(cameraRotation, -glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraRotation = glm::rotate(cameraRotation, glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraRotation = glm::rotate(cameraRotation, -glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        distanceFromCenter *= 0.9;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        distanceFromCenter *= 1.1;

    glm::vec3 cameraPos = glm::vec3(cameraRotation * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)) * distanceFromCenter;

    // std::cout << cameraPos[0] << ", " << cameraPos[1] << ", " << cameraPos[2] << std::endl;
    glm::vec3 up = glm::vec3(cameraRotation[1]);
    view = glm::lookAt(cameraPos, glm::vec3(0.0f), up);
};

OpenGLHandler::OpenGLHandler()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Mesh Visualizer", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }

    shader.Create("../res/shaders/occlusionFinder.vert", "../res/shaders/occlusionFinder.frag");

    glEnable(GL_DEPTH_TEST);
}

OpenGLHandler::~OpenGLHandler()
{
    glfwTerminate();
}

void OpenGLHandler::AddMesh(OpenGLMesh mesh)
{
    meshes.push_back(OpenGLMesh(mesh));
}

void OpenGLHandler::AddModel(Model model)
{
    models.push_back(model);
}

void OpenGLHandler::AddBoundingBox(ShapeDescriptor::cpu::Mesh mesh)
{
    boundingBoxes.push_back(MeshFunctions::boundingBox(mesh));
}

void OpenGLHandler::Draw()
{

    Shader lineShader;
    lineShader.Create("../res/shaders/line.vert", "../res/shaders/line.frag");

    shader.Use();

    int transformationLoc = glGetUniformLocation(shader.GetId(), "transformation");
    int lineTransformationLoc = glGetUniformLocation(lineShader.GetId(), "transformation");
    // int modelLoc = glGetUniformLocation(shader.GetId(), "model");
    // int viewLoc = glGetUniformLocation(shader.GetId(), "view");
    // int colorLoc = glGetUniformLocation(shader.GetId(), "uColor");
    // glUniform3f(colorLoc, 1.0f, 0.5f, 0.5f);

    float testVerts[] = {
        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, 0.5f,

        -0.5f, 0.5f, -0.5f,
        -0.5f, 0.5f, 0.5f,

        -0.5f, -0.5f, -0.5f,
        -0.5f, 0.5f, -0.5f,

        -0.5f, -0.5f, 0.5f,
        -0.5f, 0.5f, 0.5f,

        0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, 0.5f,

        0.5f, 0.5f, -0.5f,
        0.5f, 0.5f, 0.5f,

        0.5f, -0.5f, -0.5f,
        0.5f, 0.5f, -0.5f,

        0.5f, -0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,

        -0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, -0.5f,

        -0.5f, 0.5f, -0.5f,
        0.5f, 0.5f, -0.5f,

        -0.5f, -0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,

        -0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f};

    auto boundMesh = OpenGLMesh(testVerts, sizeof(testVerts));

    projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //
        auto VP = projection * view;
        // glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        // // glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        // glUniformMatrix4fv(transformationLoc, 1, GL_FALSE, glm::value_ptr(MVP));

        shader.Use();
        // for(auto mesh : meshes)
        // {
        //     auto MVP = VP * glm::translate(glm::mat4(1.0f), mesh.GetPosition());
        //     glUniformMatrix4fv(transformationLoc, 1, GL_FALSE, glm::value_ptr(MVP));
        //     mesh.Draw();
        // }

        for (auto model : models)
        {
            auto MVP = VP * glm::translate(glm::mat4(1.0f), model.GetPosition());
            glUniformMatrix4fv(transformationLoc, 1, GL_FALSE, glm::value_ptr(MVP));
            model.Draw();
        }

        lineShader.Use();

        for (auto tree : boundTrees)
        {
            currentBoundTreePointer = &tree;
            DrawLeafBoundingBoxes(tree.root, VP, lineTransformationLoc, boundMesh);
            currentBoundTreePointer = NULL;

        }

        // 
        // {
        //     auto MVP = VP * glm::translate(glm::mat4(1.0f), model.GetBoundCenter() + model.GetPosition()) * glm::scale(glm::mat4(1.0f), model.GetBoundSpan());
        //     glUniformMatrix4fv(lineTransformationLoc, 1, GL_FALSE, glm::value_ptr(MVP));
        //     boundMesh.Draw();
        // }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void OpenGLHandler::CreateMeshFromVisibleTriangles()
{

    // Copy of Bart's

    // const unsigned int offscreenTextureWidth = 4 * 7680;
    // const unsigned int offscreenTextureHeight = 4 * 4230;

    offscreenTextureWidth = 4 * 1536;
    offscreenTextureHeight = 4 * 846;

    unsigned int fbo = CreateAndBindFrameBuffer();
    unsigned int texture = CreateAndBindFrameBufferTexture();
    unsigned int rbo = CreateAndBindFrameBufferRenderBuffer();

    glfwPollEvents();
    glfwSwapBuffers(window);

    PrepareOcclusionDetectionRenderBuffer(fbo);
    SetupOcclusionDetectionShader();

    // draw

    auto mesh = ShapeDescriptor::utilities::loadOBJ("../objects/T100.obj", true);
    auto openglMesh = OpenGLMesh(mesh);
    openglMesh.Draw();

    std::cout << "Object Drawn" << std::endl;

    //

    std::vector<unsigned char> localFramebufferCopy(3 * offscreenTextureWidth * offscreenTextureHeight);

    CopyTextureToLocalBuffer(texture, localFramebufferCopy);

    std::cout << "Finding visible triangles" << std::endl;

    // Find visible traingles

    std::vector<bool> triangleAppearsInImage(mesh.vertexCount / 3);

    CheckIfTriangleAppearsInImage(localFramebufferCopy, triangleAppearsInImage);

    std::cout << "Finding visible triangles Done" << std::endl;

    //

    ShapeDescriptor::cpu::Mesh outMesh(mesh.vertexCount);

    ConstructMeshFromVisibleTriangles(mesh, outMesh, triangleAppearsInImage);

    std::cout << "New mesh computed" << std::endl;

    //

    ShapeDescriptor::dump::mesh(outMesh, "../objects/OCCLUDED2.obj");
    ShapeDescriptor::free::mesh(outMesh);
}

void OpenGLHandler::DrawLeafBoundingBoxes(BoundingBoxUtilities::BoundingBoxNode *node, glm::mat4 &VP, int lineTransformationLoc, OpenGLMesh &boundMesh)
{
    if (node->left != NULL)
        DrawLeafBoundingBoxes(node->left, VP, lineTransformationLoc, boundMesh);
    if (node->right != NULL)
        DrawLeafBoundingBoxes(node->right, VP, lineTransformationLoc, boundMesh);
    if (node->left == NULL && node->right == NULL)
    {

        auto spanf = currentBoundTreePointer->scale * node->boundingBox.span();
        auto centerf = currentBoundTreePointer->scale * node->boundingBox.center() + currentBoundTreePointer->translation;

        auto span = glm::vec3(spanf.x, spanf.y, spanf.z);
        auto center = glm::vec3(centerf.x, centerf.y, centerf.z);

        auto MVP = VP * glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), span);
        glUniformMatrix4fv(lineTransformationLoc, 1, GL_FALSE, glm::value_ptr(MVP));
        boundMesh.Draw();
    }
}

void OpenGLHandler::AddBoundTree(BoundingBoxUtilities::BoundingBoxTree &tree)
{
    boundTrees.push_back(tree);
}

unsigned int OpenGLHandler::CreateAndBindFrameBuffer()
{
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);

    return fbo;
}

unsigned int OpenGLHandler::CreateAndBindFrameBufferTexture()
{
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, offscreenTextureWidth, offscreenTextureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    return texture;
}

unsigned int OpenGLHandler::CreateAndBindFrameBufferRenderBuffer()
{
    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, offscreenTextureWidth, offscreenTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    return rbo;
}

void OpenGLHandler::PrepareOcclusionDetectionRenderBuffer(unsigned int fbo)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLHandler::SetupOcclusionDetectionShader()
{
    shader.Use();

    glm::mat4 view = glm::lookAt(occlusionDetectionCameraPosition, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 transformation = projection * view;
    int transformationLoc = glGetUniformLocation(shader.GetId(), "transformation");
    glUniformMatrix4fv(transformationLoc, 1, GL_FALSE, glm::value_ptr(transformation));
}

void OpenGLHandler::CopyTextureToLocalBuffer(unsigned int texture, std::vector<unsigned char> &localFramebufferCopy)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glReadPixels(0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_RGB, GL_UNSIGNED_BYTE, localFramebufferCopy.data());
}

void OpenGLHandler::CheckIfTriangleAppearsInImage(std::vector<unsigned char> &localFramebufferCopy, std::vector<bool> &triangleAppearsInImage)
{
    for (size_t pixel = 0; pixel < offscreenTextureWidth * offscreenTextureHeight; pixel++)
    {
        unsigned int triangleIndex =
            (((unsigned int)localFramebufferCopy.at(3 * pixel + 0)) << 16U) |
            (((unsigned int)localFramebufferCopy.at(3 * pixel + 1)) << 8U) |
            (((unsigned int)localFramebufferCopy.at(3 * pixel + 2)) << 0U);

        if (triangleIndex == 0x00FFFFFF)
            continue;

        triangleAppearsInImage.at(triangleIndex) = true;
    }
}

void OpenGLHandler::ConstructMeshFromVisibleTriangles(ShapeDescriptor::cpu::Mesh mesh, ShapeDescriptor::cpu::Mesh outMesh, std::vector<bool> &triangleAppearsInImage)
{
    unsigned int visibleVertexCount = 0;
    for (unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++)
    {
        if (triangleAppearsInImage.at(triangle))
        {
            outMesh.vertices[visibleVertexCount + 0] = mesh.vertices[3 * triangle + 0];
            outMesh.vertices[visibleVertexCount + 1] = mesh.vertices[3 * triangle + 1];
            outMesh.vertices[visibleVertexCount + 2] = mesh.vertices[3 * triangle + 2];

            outMesh.normals[visibleVertexCount + 0] = mesh.normals[3 * triangle + 0];
            outMesh.normals[visibleVertexCount + 1] = mesh.normals[3 * triangle + 1];
            outMesh.normals[visibleVertexCount + 2] = mesh.normals[3 * triangle + 2];

            visibleVertexCount += 3;
        }
    }

    std::cout << "Visible count: " << visibleVertexCount << std::endl;
}