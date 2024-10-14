#pragma once

#include <iostream>
#include <vector>

namespace Points {

template <typename T>
class PointList {
   private:
    int numPoints;   // Number of points
    int dimensions;  // Number of dimensions per point

   public:
    std::vector<T> values;  // Array of values (point data)
    // Constructor
    PointList(std::vector<T>&& values, int numPoints, int dimensions)
        : values(std::move(values)), numPoints(numPoints), dimensions(dimensions) {
        std::cout << "PointList created" << std::endl;
    }

    // Destructor to free allocated memory
    ~PointList() { std::cout << "PointList destroyed" << std::endl; }

    // Getter for number of points
    int getNumPoints() const { return numPoints; }

    // Getter for dimensions per point
    int getDimensions() const { return dimensions; }

    // Getter for the values
    std::vector<T> getValues() const { return values; }
};
}  // namespace Points
