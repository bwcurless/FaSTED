/******************************************************************************
 * File:             FilePointGenerator.hpp
 *
 * Author:           Brian Curless
 * Created:          12/11/24
 * Description:      Reads points line by line from a file.
 *****************************************************************************/

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

#include "PointGenerator.hpp"

namespace SimSearch {
class FilePointGenerator : public PointGenerator {
   public:
    FilePointGenerator(const std::string filepath, char delimeter = ',')
        : filepath(filepath), file(filepath), delimeter(delimeter) {
        if (filepath.empty()) {
            throw std::invalid_argument("filepath must be set.");
        }

        if (!file.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filepath);
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

    std::string getName() override {
        // Find the last slash in the path
        size_t lastSlashPos = filepath.find_last_of("/\\");
        std::string filename = filepath.substr(lastSlashPos + 1);

        // Find the last dot in the filename
        size_t lastDotPos = filename.find_last_of(".");
        std::string filenameWithoutExtension = filename.substr(0, lastDotPos);

        return filenameWithoutExtension;
    }

   private:
    std::string filepath;
    std::ifstream file;
    char delimeter;
};

}  // namespace SimSearch
