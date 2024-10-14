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

template <typename T>
class PointListBuilder {
   private:
    int maxPoints;

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

    // Dumb method to return 0 and handle the case where we are using half precision.
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

    /**
     * Reads a point list in from file. Pads its dimensions to the next multiple of
     * strideFactor, and the number of points up to numPointsFactor with 0's.
     *
     * \param filename The name of the file to read in.
     * \param delimeter The delimeter used between two points
     * \param strideFactor The factor to increase the dimensionality to
     * \param numPointsFactor The factor to increase the number of points to.
     *
     * \returns The list of points, padded according to the input.
     * */
    PointList<T> buildFromAsciiFile(const std::string& filename, char delimeter, int strideFactor,
                                    int numPointsFactor) {
        std::vector<T> points;
        if (filename.empty()) {
            throw std::invalid_argument("Filename must be set.");
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filename);
        }

        // Temporary variables for reading points
        std::string line;
        int numPoints = 0;
        int count = 0;

        // Read the file line by line
        while (std::getline(file, line) && maxPointsNotExceeded(numPoints)) {
            std::stringstream lineStream(line);
            count = 0;

            std::string value;
            // Parse each value on the current line
            while (std::getline(lineStream, value, delimeter)) {
                T dimension = stringToNumber(value);
                points.push_back(dimension);
                ++count;
            }
            // Pad up to the next multiple of dimensions
            while (count % strideFactor != 0) {
                T dimension = getZero();
                points.push_back(dimension);
                ++count;
            }
            ++numPoints;
        }
        int paddedDimensions = count;
        count = 0;
        // Pad up to the next multiple of numPoints Factor
        while (numPoints % numPointsFactor != 0) {
            while (count < paddedDimensions) {
                T dimension = getZero();
                points.push_back(dimension);
                count++;
            }
            numPoints++;
        }

        file.close();

        if (numPoints == 0) {
            throw std::runtime_error("No points found in file.");
        }

        // Return a PointList object constructed with the read data
        return PointList<T>(std::move(points), numPoints, count);
    }

    PointList<T> buildFromBinaryFile(const std::string& fname) {
        throw std::runtime_error("Not implemented");
    }
};
}  // namespace Points
