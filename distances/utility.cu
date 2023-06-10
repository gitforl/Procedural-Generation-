#include "utility.cuh"

void showHistogram(float *results, size_t count, size_t rangeCount, float max, float min, std::string outputDir)
{
    std::vector<int> columns(rangeCount, 0);

    float columnRange = (max - min) / rangeCount; 

    for(unsigned int i = 0; i < count; i++)
    {
        size_t column = int((results[i] - min) / columnRange);
        column = column < rangeCount ? column : rangeCount - 1;
        columns.at(column) += 1;
    }

    std::cout << "column count: " << std::endl;

    for(unsigned int i = 0; i < rangeCount; i++)
        std::cout << "column range (" << columnRange * i << ", " << columnRange * (i+1) << "): "  << columns[i] << ", " << std::endl;


    if(!outputDir.empty())
    {
        json jsonfile;

        jsonfile["binValues"] = columns;

        std::ofstream file(outputDir);
        file << jsonfile;
    }
}