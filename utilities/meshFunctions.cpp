#include <utilities/meshFunctions.hpp>


UIntVector MeshFunctions::FindSimilarVerticesIndices(unsigned int targetIndex, ShapeDescriptor::cpu::Mesh * mesh){

    ShapeDescriptor::cpu::float3 targetVertex = mesh->vertices[targetIndex];

    UIntVector similarIndices;

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        if(mesh->vertices[i] == targetVertex && i != targetIndex)
            similarIndices.push_back(i);
    }

    return similarIndices;
}

void MeshFunctions::MoveVertexAlongNormal(ShapeDescriptor::cpu::Mesh * mesh){

    float magnitude = 50;

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        mesh->vertices[i] += mesh->normals[i] * magnitude;
    }
    
}

StringUIntMap MeshFunctions::MapVertexIndices(ShapeDescriptor::cpu::Mesh * mesh){
    
    StringUIntMap indexMap;

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        indexMap[mesh->vertices[i].to_string()].push_back(i);
    }

    return indexMap;
}

StringFloat3Map MeshFunctions::VertexToAverageNormalMap(ShapeDescriptor::cpu::Mesh &mesh, StringUIntMap &indexMap){
    
    StringFloat3Map normalMap;

    for(const auto & [key, indices]: indexMap){
        
        ShapeDescriptor::cpu::float3 averageNormal;

        for(auto index: indices){
            averageNormal += mesh.normals[index];
        }

        averageNormal = averageNormal / indices.size();
        normalMap[key] = averageNormal;
    }

    return normalMap;
}

void MeshFunctions::MoveVerticesAlongAverageNormal(ShapeDescriptor::cpu::Mesh * mesh, StringUIntMap &indexMap, float maxDistance){
    
    std::cout << "current range for normal movement: [-" << maxDistance << ", " << maxDistance << "]" << std::endl;  

    for(const auto & [key, indices]: indexMap){
        
        ShapeDescriptor::cpu::float3 averageNormal;

        for(auto index: indices){
            averageNormal += mesh->normals[index];
        }

        averageNormal = averageNormal / indices.size();

        
        float distance = (((rand() % 1024) - 512) * maxDistance) / 128;

        for(auto index: indices){
            mesh->vertices[index] += averageNormal * distance;
        }
    }

}

void MeshFunctions::RecomputeVertices(ShapeDescriptor::cpu::Mesh &mesh, StringFloat3Map &normalMap){
    
    mesh.vertexCount = 3 * 4000;

    ShapeDescriptor::cpu::float3* meshVertexBuffer = new ShapeDescriptor::cpu::float3[4 * mesh.vertexCount];
    ShapeDescriptor::cpu::float3* meshNormalBuffer = new ShapeDescriptor::cpu::float3[4 * mesh.vertexCount];


    unsigned int max = mesh.vertexCount / 3;


    for(unsigned int i = 0; i < max; i++){
        
        auto first = mesh.vertices[i * 3];
        auto second = mesh.vertices[i * 3 + 1];
        auto third = mesh.vertices[i * 3 + 2];

        auto firstSecondPosition = (first + second) / 2
            + mesh.normals[i] * 
            length(second - first)/2 * (
                1 - dot(
                    normalMap[first.to_string()],
                    normalMap[second.to_string()])
                );

        auto firstThirdPosition = (first + third) / 2
            + mesh.normals[i] * 
            length(third - first)/2 * (
                1 - dot(
                    normalMap[first.to_string()],
                    normalMap[third.to_string()])
                );

        auto secondThirdPosition = (second + third) / 2
            + mesh.normals[i] * 
            length(third - second)/2 * (
                1 - dot(
                    normalMap[second.to_string()],
                    normalMap[third.to_string()])
                );


        meshVertexBuffer[i * 12] = first;
        meshVertexBuffer[i * 12 + 1] = firstSecondPosition;
        meshVertexBuffer[i * 12 + 2] = firstThirdPosition;

        auto normalOne = computeTriangleNormal(first, firstSecondPosition, firstThirdPosition);

        meshNormalBuffer[i * 12] = normalOne;
        meshNormalBuffer[i * 12 + 1] = normalOne;
        meshNormalBuffer[i * 12 + 2] = normalOne;
        
        meshVertexBuffer[i * 12 + 3] = firstSecondPosition;
        meshVertexBuffer[i * 12 + 4] = second;
        meshVertexBuffer[i * 12 + 5] = secondThirdPosition; 

        auto normalTwo = computeTriangleNormal(firstSecondPosition, second, secondThirdPosition);

        meshNormalBuffer[i * 12 + 3] = normalTwo;
        meshNormalBuffer[i * 12 + 4] = normalTwo;
        meshNormalBuffer[i * 12 + 5] = normalTwo;

        meshVertexBuffer[i * 12 + 6] = firstThirdPosition;
        meshVertexBuffer[i * 12 + 7] = firstSecondPosition;
        meshVertexBuffer[i * 12 + 8] = secondThirdPosition; 

        auto normalThree = computeTriangleNormal(firstThirdPosition, firstSecondPosition, secondThirdPosition);

        meshNormalBuffer[i * 12 + 6] = normalThree;
        meshNormalBuffer[i * 12 + 7] = normalThree;
        meshNormalBuffer[i * 12 + 8] = normalThree;        
        
        meshVertexBuffer[i * 12 + 9] = firstThirdPosition;
        meshVertexBuffer[i * 12 + 10] = secondThirdPosition;
        meshVertexBuffer[i * 12 + 11] = third; 

        auto normalFour = computeTriangleNormal(firstThirdPosition, secondThirdPosition, third);

        meshNormalBuffer[i * 12 + 9] = normalFour;
        meshNormalBuffer[i * 12 + 10] = normalFour;
        meshNormalBuffer[i * 12 + 11] = normalFour;   
   
    }

    mesh.vertices = meshVertexBuffer;
    mesh.normals = meshNormalBuffer;
    mesh.vertexCount *= 4;

}
