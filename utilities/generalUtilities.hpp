#pragma once

#include <string>
#include <vector>
#include <iostream>

#include <glm/glm.hpp>
#include <shapeDescriptor/cpu/types/float3.h>

namespace GeneralUtilities{

    template <typename T>
    void printVector(std::vector<T> vector, std::string headerText = ""){
        std::cout << headerText << std::endl;
        for (T element : vector)
        {
            std::cout << element << std::endl;
        }
    }

}