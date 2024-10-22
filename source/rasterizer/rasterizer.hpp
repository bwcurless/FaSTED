#pragma once

#include <iostream>
#include <vector>

#include "../matrix.cuh"
#include "../utils.cuh"

namespace Raster {

using Coordinate = matrix::Coordinate;

/** Rasterize a single chunk of a specified shape, based off a given upper left base coordinate.
 * Apply this chunk to the end of the input.
 *
 * \param baseCoord The upper left coordinate of this chunk.
 * \param rastered The vector to append the rastered results to.
 * \param chunkWidth How many elements wide the chunk is.
 * \param chunkHeight How many elements high the chunk is.
 */
void RasterizeChunk(const Coordinate baseCoord, std::vector<Coordinate>& rastered,
                    const int chunkWidth, const int chunkHeight) {
    for (int row = 0; row < chunkHeight; row++) {
        for (int col = 0; col < chunkWidth; col++) {
            rastered.push_back(baseCoord + Coordinate(row, col));
        }
    }
}

/** Compute the rasterized thread block layout to increase cache locality and data reuse.
 *
 *
 * \param waveWidth The width of the rasterized chunks.
 * \param waveHeight The height of the rasterized chunks.
 * \param numBlocksRows How many rows of blocks there are total in the computation.
 * \param numBlocksCols How many cols of blocks there are total in the computation.
 *
 * \return The rasterized layout.
 */
std::vector<Coordinate> RasterizeLayout(int waveWidth, int waveHeight, int numBlocksRows,
                                        int numBlocksCols) {
    std::vector<Coordinate> coords;
    coords.reserve(numBlocksRows * numBlocksCols);

    Coordinate currentCoordinate(0, 0);

    int columnsLeft = numBlocksCols;
    int rowsLeft = numBlocksRows;
    // While there are still coordinates left to raster
    while (columnsLeft && rowsLeft) {
        // Shrink the chunk dimensions at the boundaries
        int chunkWidth = std::min(waveWidth, columnsLeft);
        int chunkHeight = std::min(waveHeight, rowsLeft);

        // Rasterize this segment, push onto vector
        RasterizeChunk(currentCoordinate, coords, chunkWidth, chunkHeight);

        columnsLeft -= chunkWidth;

        // Go to next column
        if (columnsLeft != 0) {
            currentCoordinate.col += chunkWidth;
        }
        // No more columns left, go to next row
        else {
            rowsLeft -= chunkHeight;
            currentCoordinate.row += chunkHeight;
            // Make sure we aren't completed before proceeding to next line
            if (rowsLeft) {
                // Reset to first column
                columnsLeft = numBlocksCols;
                currentCoordinate.col = 0;
            }
        }
    }

    return coords;
}

/** Generate the conventional layout rasterizing entire rows at a time. Have each block compute a
 * piece of the output matrix row by row by row, working their way left to right.
 *
 * \param numBlocksRows How many rows of blocks there are total in the computation.
 * \param numBlocksCols How many cols of blocks there are total in the computation.
 *
 * \returns The layout
 */
std::vector<Coordinate> ConventionalLayout(int numBlocksRows, int numBlocksCols) {
    std::vector<Coordinate> coords;
    coords.reserve(numBlocksRows * numBlocksCols);

    for (int row = 0; row < numBlocksRows; ++row) {
        for (int col = 0; col < numBlocksCols; ++col) {
            coords.push_back({row, col});
        }
    }

    return coords;
}
}  // namespace Raster
