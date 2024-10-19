#pragma once

#include <iostream>
#include <vector>

namespace Raster {

struct Coordinate {
    int x;
    int y;

    // Default constructor
    Coordinate(int x = 0, int y = 0) : x(x), y(y) {}

    // Overload the + operator to add two 2D points
    Coordinate operator+(const Coordinate& other) const {
        return Coordinate(x + other.x, y + other.y);
    }

    // Overload the output stream operator to print the Coordinate
    friend std::ostream& operator<<(std::ostream& os, const Coordinate& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }
};

/** Rasterize a single chunk of a specified shape, and based off a given upper left base coordinate.
 *
 * \param baseCoord The upper left coordinate of this chunk.
 * \param rastered The vector to append the rastered results to.
 * \param chunkWidth How many elements wide the chunk is.
 * \param chunkHeight How many elements high the chunk is.
 */
void RasterizeChunk(Coordinate baseCoord, std::vector<Coordinate>& rastered, int chunkWidth,
                    int chunkHeight) {
    for (int row = 0; row < chunkHeight; row++) {
        for (int col = 0; col < chunkWidth; col++) {
            rastered.push_back(baseCoord + Coordinate(col, row));
        }
    }
}

/** Compute the rasterized thread block layout to increase cache locality and data reuse.
 *
 *
 * \param waveWidth The width of the rasterized chunks.
 * \param waveHeight The height of the rasterized chunks.
 * \param numBlocksRows How many rows there are total in the computation.
 * \param numBlocksCols How many cols there are total in the computation.
 *
 * \return The rasterized layout.
 */
std::vector<Coordinate> RasterizeLayout(int waveWidth, int waveHeight, int numBlocksRows,
                                        int numBlocksCols) {
    std::vector<Coordinate> coords;
    Coordinate currentCoordinate(0, 0);

    int columnsLeft = numBlocksCols;
    int rowsLeft = numBlocksRows;
    // While there are still coordinates left to raster
    while (columnsLeft && rowsLeft) {
        // Make sure we aren't at the ends...
        int chunkWidth = std::min(waveWidth, columnsLeft);
        int chunkHeight = std::min(waveHeight, rowsLeft);

        // Rasterize this segment, push onto vector
        RasterizeChunk(currentCoordinate, coords, chunkWidth, chunkHeight);

        columnsLeft = numBlocksCols - currentCoordinate.x - chunkWidth;
        // Compute the base coordinate for the next rasterized section

        // Go to next column
        if (columnsLeft != 0) {
            currentCoordinate.x += chunkWidth;
        }
        // No more columns left, go to next row
        else {
            rowsLeft = numBlocksRows - currentCoordinate.y - chunkHeight;
            currentCoordinate.y += chunkHeight;
            // Don't reset columnsLeft unless there are actually rows left
            if (rowsLeft) {
                // Reset to first column
                columnsLeft = numBlocksCols;
                currentCoordinate.x = 0;
            }
        }
    }

    if (true) {
        for (int row = 0; row < numBlocksRows; ++row) {
            for (int col = 0; col < numBlocksCols; ++col) {
                int index = row * numBlocksCols + col;
                std::cout << coords[index] << " ";
            }
            std::cout << std::endl;
        }
    }
    return coords;
}
}  // namespace Raster
