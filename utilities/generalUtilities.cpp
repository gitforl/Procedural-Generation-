#include "generalUtilities.hpp"

// void generalUtilities::printVector(std::vector<T> vector, std::string headerText = "")
// {
//     std::cout << headerText << std::endl;
//     for (T element : vector)
//     {
//         std::cout << element << std::endl;
//     }
// }

std::vector<size_t> GeneralUtilities::randomlyReduceIndices(size_t indexCount, size_t newCount)
{
    std::vector<size_t> indices(indexCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    indices.resize(newCount);

    return indices;
}