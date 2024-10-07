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

#include <vector>
namespace SumSqd {

/** Compute the sum of the squared dimensions for each point. Store results back to global memory.
 *
 *
 * \param points The input points.
 * \param numPoints How many points to compute the sum of.
 * \param numDimensions How many dimensions each point has.
 * \param sums Where to store the sums back to.
 *
 * \return
 */

template <typename Out>
__global__ void SquaredSumsKernel(half2* points, const int numPoints, const int numDimensions,
                                  Out* sums) {
    // Each block is responsible for reducing one point in this simple implementation
    __shared__ Out sum;

    int pointIndex = blockIdx.x;

    if (threadIdx.x == 0) {
        sum = 0.0;
    }

    __syncthreads();

    // TODO make this a proper reduction
    // Reads two half values at a time
    int normalizedDimensions = numDimensions / 2;
    int firstDimension = pointIndex * normalizedDimensions;
    for (int i = threadIdx.x; i < normalizedDimensions; i += blockDim.x) {
        half2 dims = points[firstDimension + i];
        float upCast1 = __half2float(dims.x);
        float upCast2 = __half2float(dims.y);
        Out squared = upCast1 * upCast1;
        squared += upCast2 * upCast2;
        atomicAdd(&sum, squared);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        sums[pointIndex] = sum;
    }
}

/** Given a set of points, computes the sum of each squared dimension for every point. This function
 * allocates the memory it needs to store the sums for each point. You must free it.
 *
 * \param points The points to compute the squared sums for. Located in global memory.
 // TODO come back and add missing params
 *
 * \return The sums of the squared dimensions of each point.
 */
template <typename Out>
Out* ComputeSquaredSums(half2* points, const int numPoints, const int numDimensions) {
    // Allocate memory
    Out* sums;
    cudaMalloc(&sums, numPoints * sizeof(Out));

    // Determine launch parameters
    dim3 blockDims(128);
    dim3 gridDims(numPoints);

    // Launch Kernel
    SquaredSumsKernel<<<gridDims, blockDims>>>(points, numPoints, numDimensions, sums);

    // Return results
    return sums;
}

}  // namespace SumSqd
#endif /* ifndef BLOCKSUMSQUARED */
