#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

    /** Convert a string to the specified precision value.
     *
     * \param str The string to convert to a numeric.
     */
    T stringToNumber(const std::string& str) {
        try {
            if constexpr (std::is_same<T, float>::value) {
                return std::stof(str);  // Convert string to float
            } else if constexpr (std::is_same<T, double>::value) {
                return std::stod(str);  // Convert string to double
            } else if constexpr (std::is_same<T, half>::value) {
                float temp = std::stof(str);  // Convert string to float first
                return half(temp);            // Then to half
            } else {
                throw std::invalid_argument("Unsupported type for conversion");
            }
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("Invalid string for " + std::string(typeid(T).name()) +
                                     " conversion: " + str);
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
    // Constructor to initialize the builder
    PointListBuilder() : maxPoints{0} {}

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

    /** Parse an entire line of a file. Append each dimension to points vector.
     *
     *
     * \returns How many dimensions were read
     */
    int parseLine(std::string line, char delimeter, std::vector<T>& points) {
        int dimCount = 0;
        std::stringstream lineStream(line);
        std::string value;

        // Parse each value on the current line
        while (std::getline(lineStream, value, delimeter)) {
            T dimension = stringToNumber(value);
            points.push_back(dimension);
            ++dimCount;
        }

        return dimCount;
    }

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

    /**
     * Reads a point list in from file. Pads its dimensions to the next multiple of
     * strideFactor, and the number of points up to numPointsFactor with 0's.
     *
     * \param filename The name of the file to read in.
     * \param delimeter The delimeter used between two points
     * \param dimensionFactor The factor to increase the dimensionality to
     * \param numPointsFactor The factor to increase the number of points to.
     *
     * \returns The list of points, padded according to the input.
     * */
    PointList<T> buildFromAsciiFile(const std::string& filename, char delimeter = ',',
                                    int dimensionFactor = 1, int numPointsFactor = 1) {
        std::vector<T> points;
        if (filename.empty()) {
            throw std::invalid_argument("Filename must be set.");
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filename);
        }

        bool firstIter = true;
        int numDimensions;
        int paddedDimensions;

        // Read the file line by line
        int pointCount = 0;
        std::string line;
        while (std::getline(file, line) && maxPointsNotExceeded(pointCount)) {
            int dimCount = parseLine(line, delimeter, points);

            // Check actual number of dimensions parsed
            if (firstIter) {
                firstIter = false;
                // First row in file defines how many dimemsions to expect on subsequent rows
                numDimensions = dimCount;
                // Compute the total dimensions needed with padding
                paddedDimensions = roundToNearestMultiple(numDimensions, dimensionFactor);
            } else if (dimCount != numDimensions) {
                throw std::runtime_error(
                    "Dimensions on subsequent lines didn't match the first line");
            }

            zeroPadPoint(points, paddedDimensions - numDimensions);

            ++pointCount;
        }
        // Add extra points of all 0's
        int paddedPoints = roundToNearestMultiple(pointCount, numPointsFactor);
        int numZerosToAppend = paddedPoints * paddedDimensions;
        zeroPadPoint(points, numZerosToAppend);

        file.close();

        if (pointCount == 0) {
            throw std::runtime_error("No points found in file.");
        }

        // Return a PointList object constructed with the read data
        return PointList<T>(std::move(points), paddedPoints, pointCount, paddedDimensions,
                            numDimensions);
    }
};
}  // namespace Points
