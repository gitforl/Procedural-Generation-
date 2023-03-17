#include "mathUtilities.hpp"
#include <cmath>

float computeFloatAverage(std::vector<float> values){
    float distanceSum = 0.0f;

    for (float value:values)
        distanceSum += value;
    
    return (distanceSum / values.size());
}

float computeFloatStandardDeviation(std::vector<float> values, float average){
    float deviationSum = 0.0f;

    for (float value: values)
    {
        float difference = (value - average);
        deviationSum += (difference * difference);
    }

    return sqrt(deviationSum / values.size());
}