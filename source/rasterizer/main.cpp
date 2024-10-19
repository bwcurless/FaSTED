#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "rasterizer.hpp"

int main(int argc, char* argv[]) {
    int rows = 21;
    int cols = 21;
    auto rasterized = Raster::RasterizeLayout(5, 5, rows, cols);

    // Invert the coordinates to be easier to look at
    int inverted[rows][cols];
    for (size_t i = 0; i < rows * cols; i++) {
        auto val = rasterized[i];
        inverted[val.y][val.x] = i;
    }

    // Print it out
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            std::cout << std::setw(3) << inverted[row][col] << " ";
        }
        std::cout << std::endl;
    }

    return 0;  // Return 0 to indicate success
}
