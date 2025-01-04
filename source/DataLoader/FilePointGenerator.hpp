/******************************************************************************
 * File:             FilePointGenerator.hpp
 *
 * Author:           Brian Curless
 * Created:          12/11/24
 * Description:      Reads points line by line from a file.
 *****************************************************************************/

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

#include "PointGenerator.hpp"

namespace SimSearch {
class FilePointGenerator : public PointGenerator {
   public:
    FilePointGenerator(const std::string filename, char delimeter = ',')
        : filename(filename), file(filename), delimeter(delimeter) {
        if (filename.empty()) {
            throw std::invalid_argument("Filename must be set.");
        }

        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filename);
        }
    }

    ~FilePointGenerator() { file.close(); }

    std::optional<std::vector<double>> next() override {
        std::vector<double> point;

        // Read next line from file
        // If at end of file, exit early
        std::string line;
        if (!std::getline(file, line)) {
            return std::nullopt;
        }
        std::stringstream lineStream(line);

        // Parse each value on the current line
        std::string value;
        while (std::getline(lineStream, value, delimeter)) {
            try {
                double dimension = std::stod(value);
                point.push_back(dimension);
            } catch (const std::exception& e) {
                std::cout << "Failed to parse value: " << value << " from dataset" << std::endl;
                throw;
            }
        }
        return point;
    }

    std::string getName() override { return filename; }

   private:
    std::string filename;
    std::ifstream file;
    char delimeter;
};

}  // namespace SimSearch
