#include <iostream>
#include <vector>
#include <map>
// #include <../Dissimilarity-Tree-Reproduction/input/download/SHREC2016/SHREC2016_partial_retrieval/complete_objects
// #include <../libShapeDescriptor/src/shapeDescriptor/cpu/types/Mesh.h>
#include <filesystem>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>


std::vector<unsigned int> FindSimilarVerticesIndices(unsigned int targetIndex, ShapeDescriptor::cpu::Mesh * mesh){

    ShapeDescriptor::cpu::float3 targetVertex = mesh->vertices[targetIndex];

    std::vector<unsigned int> similarIndices;

    //j4irhr

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        if(mesh->vertices[i] == targetVertex && i != targetIndex)
            similarIndices.push_back(i);
    }

    return similarIndices;
}

void MoveVertexAlongNormal(ShapeDescriptor::cpu::Mesh * mesh){

    float magnitude = 50;

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        mesh->vertices[i] += mesh->normals[i] * magnitude;
    }
    
}


std::map<std::string, std::vector<unsigned int>> MapVertexIndices(ShapeDescriptor::cpu::Mesh * mesh){
    
    std::map<std::string, std::vector<unsigned int>> indexMap;

    for(unsigned int i = 0; i < mesh->vertexCount; i++){
        indexMap[mesh->vertices[i].to_string()].push_back(i);
    }

    return indexMap;
}

template <typename T>
void printVector(std::vector<T> vector, std::string headerText = ""){
    std::cout << headerText << std::endl;
    for(T element: vector){
        std::cout << element << std::endl;
    }
}

int main()
{

    // const std::string objSrcPath = "../Dissimilarity-Tree-Reproduction/input/download/SHREC2016/SHREC2016_partial_retrieval/complete_objects/T100.obj";
    const std::string objSrcPath = "../objects/TN100.obj";

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadOBJ(objSrcPath, true);

    // auto similarVertices = FindSimilarVerticesIndices(200, &mesh);

    // // MoveVertexAlongNormal(&mesh);
    // for(auto i : similarVertices){
    //     std::cout << i << std::endl;
    // }
   
    auto vertexMap = MapVertexIndices(&mesh);
    std::cout << "Map entries: " << vertexMap.size() << std::endl;

    auto mapElement = vertexMap[mesh.vertices[200].to_string()];
    printVector(mapElement, "elements in " + mesh.vertices[200].to_string());

    const std::filesystem::path outpath = "../objects/out.obj";
    ShapeDescriptor::dump::mesh(mesh, outpath);

    std::cout << "Object vertexCount: " << mesh.vertexCount << std::endl;


}