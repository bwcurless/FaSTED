/******************************************************************************
 * File:             utils.cuh
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      General purpose utility functions and constants
 *****************************************************************************/

#ifndef UTILS_CUH_JAUS2UK0
#define UTILS_CUH_JAUS2UK0

#include <iostream>

#define WARPSIZE 32
constexpr int SharedMemWidth = 128;  // 128 bytes, 32 lanes of 4 bytes each
constexpr bool Debug = false;        // Enables some debug print statements.
constexpr bool SmallDebug =
    false;  // Only set this if you are running with small datasets. Prints out a lot of data.

__device__ __host__ void PrintMatrix(const char* name, half* matrix, int m, int n);

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/** Allocate space for a host value on the device, and copy the value to the device.
 *
 * \param devicePointer The pointer that you want to allocate space for on the device.
 * \param hostValue The value on the host that you want to copy to the memory allocated and
 * stored in devicePointer.
 *
 */
template <typename T>
inline void allocateAndTransferToDevice(T*& devicePointer, const T& hostValue) {
    gpuErrchk(cudaMalloc(&devicePointer, sizeof(hostValue)));
    gpuErrchk(cudaMemcpy(devicePointer, &hostValue, sizeof(hostValue), cudaMemcpyHostToDevice));
}

void checkLastCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

// Return the ID of the steaming multiprocesser this block is running on
__device__ uint get_smid(void) {
    uint ret;

    asm("mov.u32 %0, %smid;" : "=r"(ret));

    return ret;
}

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
template <typename T>
__device__ __host__ void PrintMatrix(const char* name, T* matrix, int m, int n) {
    printf("Printing matrix %s\n", name);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            printf("%.0f ", static_cast<double>(matrix[row * n + col]));
        }
        printf("\n");
    }
}

#endif /* end of include guard: UTILS_CUH_JAUS2UK0 */
