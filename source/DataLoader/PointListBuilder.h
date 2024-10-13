#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PointList.h"
#include "half.hpp"

namespace Points {

using half = half_float::half;

template <typename T>
class PointListBuilder {
   private:
    // Private method to convert string to float, double, or half
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

   public:
    // Constructor to initialize the builder
    PointListBuilder() {}

    PointList<T> buildFromAsciiFile(const std::string& filename, char delimeter) {
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
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            count = 0;

            std::string value;
            // Parse each value on the current line
            while (std::getline(lineStream, value, delimeter)) {
                T dimension = stringToNumber(value);
                points.push_back(dimension);
                ++count;
            }

            ++numPoints;
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
