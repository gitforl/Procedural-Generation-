#pragma once

#include <string>
#include <vector>
#include <iostream>

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