#include "openglMesh.hpp"



OpenGLMesh::OpenGLMesh(ShapeDescriptor::cpu::Mesh &mesh, meshTypes type)

{

    OpenGLMesh::type = type;

    if(type == meshTypes::Occlusion)
        occlusionMeshInit(mesh);
    else if(type == meshTypes::Scaled)
        simpleMeshInit(mesh, 0.01f);
    else
        simpleMeshInit(mesh, 1.0f);

}

OpenGLMesh::OpenGLMesh(float* vertices, unsigned int size){

    OpenGLMesh::type = meshTypes::Lines;
    vertexMeshInit(vertices, size);
}

void OpenGLMesh::Draw(){
    glBindVertexArray(VAO);
    if(type == meshTypes::Lines)
        glDrawArrays(GL_LINES, 0, vertexCount);
    else
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}

// void OpenGLMesh::UniformModel(unsigned int uniformLocation)
// {
//     glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
//     glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, glm::value_ptr(model));
// }
// void OpenGLMesh::SetPosition(glm::vec3 newPosition)
// {
//     position = newPosition;
// }

// glm::vec3 OpenGLMesh::GetPosition()
// {
//     return position;
// }

meshTypes OpenGLMesh::GetType(){
    return type;
}

void OpenGLMesh::occlusionMeshInit(ShapeDescriptor::cpu::Mesh &mesh)
{
    vertexCount = mesh.vertexCount;
    unsigned int triangleCount = vertexCount / 3;

    float data[mesh.vertexCount * 6];

    float scaler = 0.01;

    for(unsigned int i = 0; i < triangleCount; i++)
    {
        float red = float((i & 0x00FF0000U) >> 16U) / 255.0f;
        float green = float((i & 0x0000FF00U) >> 8U) / 255.0f;
        float blue = float((i & 0x000000FFU) >> 0U) / 255.0f;

        for(unsigned int j = 0; j < 3; j++)
        {
            data[i * 18 + j * 6 + 0] = scaler * mesh.vertices[i * 3 + j].x;
            data[i * 18 + j * 6 + 1] = scaler * mesh.vertices[i * 3 + j].y;
            data[i * 18 + j * 6 + 2] = scaler * mesh.vertices[i * 3 + j].z;

            data[i * 18 + j * 6 + 3] = red;
            data[i * 18 + j * 6 + 4] = green;
            data[i * 18 + j * 6 + 5] = blue;
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void OpenGLMesh::simpleMeshInit(ShapeDescriptor::cpu::Mesh &mesh, float scaler){
    vertexCount = mesh.vertexCount;

    float vertices[mesh.vertexCount * 6];

    for(int i = 0; i < mesh.vertexCount; i++)
    {
        vertices[i * 6 + 0] = scaler * mesh.vertices[i].x;
        vertices[i * 6 + 1] = scaler * mesh.vertices[i].y;
        vertices[i * 6 + 2] = scaler * mesh.vertices[i].z;

        vertices[i * 6 + 3] = scaler * mesh.normals[i].x;
        vertices[i * 6 + 4] = scaler * mesh.normals[i].y;
        vertices[i * 6 + 5] = scaler * mesh.normals[i].z;
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void OpenGLMesh::vertexMeshInit(float* vertices, unsigned int size){
    if(size%6 != 0)
        std::cout << "Not proper count of vertices to create mesh" << std::endl;

    vertexCount = size;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}
