#include <iostream>
#include <stdexcept>

#include "PointList.hpp"
#include "PointListBuilder.hpp"
#include "half.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;  // Exit with error if the number of arguments is incorrect
    }

    std::string filename = argv[1];  // Get the filename from the command-line argument

    try {
        // Attempt to build the PointList using the provided filename
        Points::PointList<half_float::half> pointList =
            Points::PointListBuilder<half_float::half>().withMaxPoints(10).buildFromAsciiFile(
                filename, ',', 64, 128);
        std::cout << pointList.getNumPoints() << std::endl;

    } catch (const std::exception& e) {
        // Handle any errors that occur during the building process
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;  // Return a non-zero value to indicate an error occurred
    }

    return 0;  // Return 0 to indicate success
}
