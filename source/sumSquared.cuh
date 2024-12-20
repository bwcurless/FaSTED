/******************************************************************************
 * File:             blockSumSquared.cuh
 *
 * Author:           Brian Curless
 * Created:          10/06/24
 * Description:      Takes in a set of points, and computes the sum of the squared dimensions.
 *			Stores the results back to global memory.
 *****************************************************************************/

#ifndef BLOCKSUMSQUARED
#define BLOCKSUMSQUARED

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <vector>

#include "utils.cuh"

namespace SumSqd {

/** Compute the sum of the squares of a half2 value. There are two values, square each one, then
 * compute the sum of them in the specified precision.
 *
 * \param value The value to square and sum.
 *
 * \return The resulting sum in the specified precision.
 */
template <typename Out>
__device__ Out SumSquareHalf2(half2 value);

template <>
__device__ float SumSquareHalf2(half2 value) {
    float2 fValues = __half22float2(value);
    fValues.x = fValues.x * fValues.x;
    fValues.y = fValues.y * fValues.y;
    return __fadd_rz(fValues.x, fValues.y);
}

/** Compute the sum of the squared dimensions for each point. Store results back to global memory.
 *
 *
 * \param points The input points.
 * \param numPoints How many points to compute the sum of across the entire grid.
 * \param numDimensions How many dimensions each point has.
 * \param sums Where to store the sums back to.
 *
 * \return
 */

template <typename Out>
__global__ void SquaredSumsKernel(half2* points, const int numPoints, const int numDimensions,
                                  Out* sums) {
    // Each block is responsible for reducing one point in this simple implementation
    // TODO Investigate if doing a shared memory reduction using round to zero addition would be
    // faster and yield the same results. Can't use atomicAdd because it rounds to nearest even.
    int pointIndex = blockIdx.x;
    Out localSum = 0;

    // Reads two half values at a time
    int normalizedDimensions = numDimensions / 2;
    int firstDimension = pointIndex * normalizedDimensions;
    for (int i = threadIdx.x; i < normalizedDimensions; i += blockDim.x) {
        half2 dims = points[firstDimension + i];
        localSum = __fadd_rz(localSum, SumSquareHalf2<Out>(dims));
    }

    if (threadIdx.x == 0) {
        sums[pointIndex] = localSum;
    }
}

/** Given a set of points, computes the sum of each squared dimension for every point. This function
 * allocates the memory it needs to store the sums for each point. You must free it.
 *
 * \param points The points to compute the squared sums for. Located in global memory.
 * \param numPoints How many points to compute the sum of squares for.
 * \param numDimensions How many dimensions there are per point.
 *
 * \return The sums of the squared dimensions of each point.
 */
template <typename Out>
Out* ComputeSquaredSums(half2* points, const int numPoints, const int numDimensions) {
    // Allocate memory
    Out* sums;
    size_t sumsSize = numPoints * sizeof(Out);
    cudaMalloc(&sums, sumsSize);

    // Determine launch parameters
    dim3 blockDims(
        1);  // Weird, but can't do an atomicSum reduction here since we need to round to zero.
    dim3 gridDims(numPoints);

    // Launch Kernel
    SquaredSumsKernel<<<gridDims, blockDims>>>(points, numPoints, numDimensions, sums);

    if (Debug) {
        Out* h_sums = static_cast<Out*>(malloc(sumsSize));

        cudaDeviceSynchronize();
        cudaMemcpy(h_sums, sums, sumsSize, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        PrintMatrix<Out>("Sums of squared elements", h_sums, numPoints, 1);
        free(h_sums);
    }

    // Return results
    return sums;
}

}  // namespace SumSqd
#endif /* ifndef BLOCKSUMSQUARED */
