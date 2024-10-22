/******************************************************************************
 * File:             matrix.cuh
 *
 * Author:           Brian Curless
 * Created:          10/21/24
 * Description:      Generally useful matrix manipulation structures.
 *****************************************************************************/
#pragma once

namespace matrix {

/** A 2 dimensional coordinate specified by a row and a column. Useful when thinking in terms of
 * matix coordinates.
 */
struct Coordinate {
    int row;
    int col;

    // Default constructor
    __device__ __host__ Coordinate(int row = 0, int col = 0) : row(row), col(col) {}

    // Overload the + operator to add two 2D points
    __device__ __host__ Coordinate operator+(const Coordinate& other) const {
        return Coordinate(row + other.row, col + other.col);
    }

    // Overload the output stream operator to print the Coordinate
    friend std::ostream& operator<<(std::ostream& os, const Coordinate& p) {
        os << "(" << p.row << ", " << p.col << ")";
        return os;
    }
};
}  // namespace matrix
