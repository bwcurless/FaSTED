#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PointGenerator.hpp"
#include "PointList.hpp"
#include "half.hpp"

namespace Points {

using half = half_float::half;

/** A way to create a PointList. Can read a series of n-dimensional points from a file, and
 * optionally apply padding to them.
 *
 */
template <typename T>
class PointListBuilder {
   private:
    int maxPoints{};
    SimSearch::PointGenerator* generator;

    /** Append a certain number of zeros to points.
     *
     * \param points Where to append the zeros to.
     * \param numValues How many zeros to append.
     */
    void zeroPadPoint(std::vector<T>& points, int numValues) {
        for (int i = 0; i < numValues; ++i) {
            T dimension = getZero();
            points.push_back(dimension);
        }
    }

    /** Given a value, round it up to the next nearest multiple of the given multiple. Useful for
     * rounding integers up.
     *
     * \param value The value to round up.
     * \param multiple The multiple to round up to.
     *
     * \returns The rounded value.
     */
    int roundToNearestMultiple(int value, int multiple) {
        return ceil(1.0 * value / multiple) * multiple;
    }

    /** Convert a value to the specified precision in the templated type.
     *
     * \param val The value to convert.
     */
    T castValue(const double val) {
        try {
            if constexpr (std::is_same<T, float>::value) {
                return static_cast<float>(val);  // Convert string to float
            } else if constexpr (std::is_same<T, double>::value) {
                return val;  // Convert string to double
            } else if constexpr (std::is_same<T, half>::value) {
                float temp = static_cast<float>(val);  // Convert string to float first
                return half(temp);                     // Then to half
            } else {
                throw std::invalid_argument("Unsupported type for conversion");
            }
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("Invalid string for " + std::string(typeid(T).name()) +
                                     " conversion: " + std::to_string(val));
        }
    }

    /** Dumb method to return 0 and handle the case where we are using half precision.
     */
    T getZero() const {
        if constexpr (std::is_same<T, half>::value) {
            return half(0);
        } else
            return 0.0;
    }

    /**
     * Check if there are more points to process.
     *
     * \param numPoints The current number of points loaded
     *
     * \returns True if the maximum number of points specified has not been exceeded.
     *
     * */
    bool maxPointsNotExceeded(const int numPoints) const {
        return (maxPoints == 0 || numPoints < maxPoints);
    }

   public:
    /** Constructor to initialize the builder
     *
     * \param generator The Point generator to use to obtain each point.
     */
    PointListBuilder(SimSearch::PointGenerator* generator) : maxPoints{0}, generator{generator} {}

    /**
     * Limit the builder to only reading a certain number of points.
     *
     * \param maxNumber How many points to read.
     *
     * \return The PointListBuilder configured with a max number of points.
     * */
    PointListBuilder& withMaxPoints(int maxNumber) {
        if (maxNumber > 0) {
            maxPoints = maxNumber;
            return *this;
        } else {
            throw std::invalid_argument("Max number of points must be greater than 0");
        }
    }

    /** Return the name of the dataset.
     */
    std::string getDatasetName() { return generator->getName(); }

    /**
     * Creates a point list. Pads its dimensions to the next multiple of
     * strideFactor, and the number of points up to numPointsFactor with 0's.
     *
     * \param dimensionFactor The factor to increase the dimensionality to
     * \param numPointsFactor The factor to increase the number of points to.
     *
     * \returns The list of points, padded according to the input.
     * */
    PointList<T> build(int dimensionFactor = 1, int numPointsFactor = 1) {
        std::vector<T> points;
        bool firstIter = true;
        int numDimensions;
        int paddedDimensions;

        // Read the file line by line
        int pointCount = 0;
        std::optional<std::vector<double>> nextPoint;
        while ((nextPoint = generator->next()) && maxPointsNotExceeded(pointCount)) {
            int dimCount = nextPoint.value().size();

            // Append point to PointList
            for (auto dim : nextPoint.value()) {
                points.push_back(castValue(dim));
            }

            // Check actual number of dimensions parsed
            if (firstIter) {
                firstIter = false;
                // First row in file defines how many dimemsions to expect on subsequent rows
                numDimensions = dimCount;
                // Compute the total dimensions needed with padding
                paddedDimensions = roundToNearestMultiple(numDimensions, dimensionFactor);
            } else if (dimCount != numDimensions) {
                throw std::runtime_error(
                    "Dimensions of subsequent points didn't match the first point");
            }

            zeroPadPoint(points, paddedDimensions - numDimensions);

            ++pointCount;
        }
        // Add extra points of all 0's
        int paddedPoints = roundToNearestMultiple(pointCount, numPointsFactor);
        int numZerosToAppend = paddedPoints * paddedDimensions;
        zeroPadPoint(points, numZerosToAppend);

        if (pointCount == 0) {
            throw std::runtime_error("No points were generated.");
        }

        // Return a PointList object constructed with the read data
        return PointList<T>(std::move(points), paddedPoints, pointCount, paddedDimensions,
                            numDimensions);
    }
};
}  // namespace Points
