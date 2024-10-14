#pragma once

#include <iostream>
#include <vector>

namespace Points {

template <typename T>
class PointList {
   private:
    int numPoints;         // Number of points, including padded ones
    int numActualPoints;   // Number of actual points read in.
    int dimensions;        // Number of dimensions per point
    int actualDimensions;  // Number of actual dimensions per point.

   public:
    std::vector<T> values;  // Array of values (point data)
    // Constructor
    PointList(std::vector<T>&& values, int numPoints, int numActualPoints, int dimensions,
              int actualDimensions)
        : numPoints(numPoints),
          numActualPoints(numActualPoints),
          dimensions(dimensions),
          actualDimensions(actualDimensions),
          values(std::move(values)) {}

    PointList() {}

    ~PointList() {}

    int getNumPoints() const { return numPoints; }

    int getActualNumPoints() const { return numActualPoints; }

    int getDimensions() const { return dimensions; }

    int getActualDimensions() const { return actualDimensions; }

    std::vector<T> getValues() const { return values; }
};
}  // namespace Points
