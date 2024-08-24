/******************************************************************************
 * File:             utils.cuh
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      General purpose utility functions and constants
 *****************************************************************************/

#ifndef UTILS_CUH_JAUS2UK0
#define UTILS_CUH_JAUS2UK0

#define WARPSIZE 32
constexpr int SharedMemWidth = 128;  // 128 bytes, 32 lanes of 4 bytes each
constexpr bool Debug = false;

__device__ __host__ void PrintMatrix(const char* name, half* matrix, int m, int n);

__device__ __host__ constexpr bool IsDivisionExact(int dividend, int divisor) {
    return dividend == (divisor * (dividend / divisor));
}

// Compile time fails if constexpr division results in any rounding
template <size_t dividend, size_t divisor>
__device__ void Assert_Is_Division_Exact() {
    // Check if the division is exact
    // Forced round down will happen here due to truncation.
    static_assert(IsDivisionExact(dividend, divisor), "Division is not exact.");
}

/** Pretty print an array as a matrix.
 *
 *
 * \param name Name of the matrix
 * \param matrix First address of matrix
 * \param m Height dimension of matrix
 * \param n Width dimension of matrix
 *
 * \return
 */
__device__ __host__ void PrintMatrix(const char* name, half* matrix, int m, int n) {
    if (Debug) {
        printf("Printing matrix %s\n", name);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                printf("%.0f ", static_cast<double>(matrix[row * n + col]));
            }
            printf("\n");
        }
    }
}

#endif /* end of include guard: UTILS_CUH_JAUS2UK0 */
