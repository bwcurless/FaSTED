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

namespace BlockSumSquared {

/** Given a set of points, computes the sum of each squared dimension for every point. This function
 * allocates the memory it needs to store the sums for each point. You must free it.
 *
 * \param points The points to compute the squared sums for. Located in global memory.
 // TODO come back and add missing params
 *
 * \return The sums of the squared dimensions of each point.
 */
template <typename SumPrecision, typename InputPrecision>
SumPrecision* ComputeSquaredSums(InputPrecision* points, const int numPoints,
                                 const int numDimensions) {
    // Allocate memory
    SumPrecision* sums;
    cudaMalloc(&sums, numPoints * sizeof(SumPrecision));

    // Determine launch parameters

    // Launch Kernel

    // Return results
    return sums;
}

}  // namespace BlockSumSquared
#endif /* ifndef BLOCKSUMSQUARED */
