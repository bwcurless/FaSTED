#pragma once

#include <iostream>
#include <vector>

namespace Points {

/** A set of n-dimensional points. Contained in a single vector.
 *
 *
 */
template <typename T>
class PointList {
   private:
    int numPaddedPoints;      // Number of points, including padded ones
    int numRealPoints;        // Number of actual points read in.
    int numPaddedDimensions;  // Number of dimensions per point, including padded ones
    int numRealDimensions;    // Number of real dimensions per point.

   public:
    std::vector<T> values;  // Array of values (point data)

    PointList(std::vector<T>&& values, int numPaddedPoints, int numRealPoints, int paddedDimensions,
              int realDimensions)
        : numPaddedPoints(numPaddedPoints),
          numRealPoints(numRealPoints),
          numPaddedDimensions(paddedDimensions),
          numRealDimensions(realDimensions),
          values(std::move(values)) {}

    PointList() {}

    int getNumPoints() const { return numPaddedPoints; }

    int getActualNumPoints() const { return numRealPoints; }

    int getDimensions() const { return numPaddedDimensions; }

    int getActualDimensions() const { return numRealDimensions; }

    std::vector<T> getValues() const { return values; }
};
}  // namespace Points
