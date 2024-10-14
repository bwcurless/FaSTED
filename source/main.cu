#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <iostream>
#include <vector>

#include "DataLoader/PointListBuilder.hpp"
#include "findPairs.cuh"
#include "sumSquared.cuh"
#include "utils.cuh"

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

// ---------- Search Parameters ----------
constexpr float epsilonSquared = 0.1;

/** Create a set of points with monatomically increasing values. Increments by 1 for every point.
 * Note that half values can only count up to about 64000, so the max value is
 * capped at 32768.
 *
 * \param values The vector to push the values onto.
 *
 */
void GenerateIncreasingPoints(std::vector<half2>& values, int numPoints, int numDimensions) {
    // Kind of a hack but we go to NaN if we let it keep incrementing
    int maxFloat = 32768;
    // Fill the vector with increasing half-precision values
    // Note that this gets funny > 2048 because of imprecision of half values
    for (int m = 0; m < numPoints; m++) {
        for (int k = 0; k < numDimensions; k += 2) {
            half2 val{};
            val.x = static_cast<half>(min(maxFloat, m * numDimensions + k));
            val.y = static_cast<half>(min(maxFloat, m * numDimensions + k + 1));
            values.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global A", reinterpret_cast<half*>(values.data()), numPoints,
                          numDimensions);
    }
}

/** Create a set of points that is similar to an identity matrix. Will be all 0's except for
 * diagonal values.
 *
 * \param values The vector to push values onto.
 *
 */
void GenerateIdentityMatrixPoints(std::vector<half2>& values, int numPoints, int numDimensions) {
    // Create identity matrix
    for (int row = 0; row < numPoints; row++) {
        for (int col = 0; col < numDimensions; col += 2) {
            half2 val{0, 0};
            if (col == row)
                val.x = 1;
            else if (col + 1 == row)
                val.y = 1;
            values.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global B", reinterpret_cast<half*>(values.data()), numPoints,
                          numDimensions);
    }
}

/** Allocates space for and transfers query and candidate points stored in host memory to global
 * memory. You must free the memory allocated by this method when you are done using it.
 *
 *
 * \param Precision The data type of each dimension.
 * \param h_Query The actual query points stored on the host.
 * \param h_Candidate The actual candidate points stored on the host.
 * \param d_Query Pointer to query points array on device.
 * \param d_Candidate Pointer to the candidate points array on device.
 *
 * \return
 */
template <typename Precision, typename T>
void TransferPointsToGMem(std::vector<Precision>& h_Query, std::vector<Precision>& h_Candidate,
                          T** d_Query, T** d_Candidate) {
    size_t querySize = h_Query.size() * sizeof(h_Query[0]);
    size_t candidateSize = h_Candidate.size() * sizeof(h_Candidate[0]);

    cudaMalloc(d_Query, querySize);
    cudaMalloc(d_Candidate, candidateSize);

    cudaMemcpy(*d_Query, h_Query.data(), querySize, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_Candidate, h_Candidate.data(), candidateSize, cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;  // Exit with error if the number of arguments is incorrect
    }

    std::string filename = argv[1];  // Get the filename from the command-line argument

    half2 *d_AValues, *d_BValues;
    // Attempt to build the PointList using the provided filename
    Mma::mmaShape bDims = BlockTile::GetBlockTileDims();
    Points::PointList<half_float::half> pointList =
        Points::PointListBuilder<half_float::half>().buildFromAsciiFile(filename, ',', bDims.k,
                                                                        bDims.m);

    Mma::mmaShape searchShape{pointList.getNumPoints(), pointList.getNumPoints(),
                              pointList.getDimensions()};

    std::cout << "M: " << searchShape.m << std::endl;
    std::cout << "N: " << searchShape.n << std::endl;
    std::cout << "K: " << searchShape.k << std::endl;

    TransferPointsToGMem(pointList.values, pointList.values, &d_AValues, &d_BValues);

    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate this just so kernel actually does something
    unsigned long long* d_iterationCount;
    unsigned long long h_iterationCount = 0;
    cudaMalloc(&d_iterationCount, sizeof(unsigned long long));
    cudaMemcpy(d_iterationCount, &h_iterationCount, sizeof(unsigned long long),
               cudaMemcpyHostToDevice);

    printf("Running kernel\n");

    // Compute sums of squared dimensions
    using sumSize = float;
    sumSize *d_ASqSums, *d_BSqSums;
    d_ASqSums = SumSqd::ComputeSquaredSums<sumSize>(d_AValues, searchShape.m, searchShape.k);
    d_BSqSums = SumSqd::ComputeSquaredSums<sumSize>(d_BValues, searchShape.n, searchShape.k);

    cudaEventRecord(start, 0);

    BlockTile::FindPairs(BlockTile::FindPairsParams{epsilonSquared, searchShape, d_iterationCount,
                                                    d_AValues, d_BValues, d_ASqSums, d_BSqSums});

    gpuErrchk(cudaEventRecord(stop, 0));

    gpuErrchk(cudaEventSynchronize(stop));

    cudaMemcpy(&h_iterationCount, d_iterationCount, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    printf("Number of total iterations: %lld\n", h_iterationCount);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= 1000;

    printf("Kernel Elapsed time: %f seconds\n", elapsedTime);
    // Estimated TFLOPS that we computed
    const float tflops =
        static_cast<float>(searchShape.m) * searchShape.n * searchShape.k * 2 / elapsedTime / 1e12;
    printf("Estimated TFLOPS %.3f\n", tflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_iterationCount);
    cudaFree(d_AValues);
    cudaFree(d_BValues);
    cudaFree(d_ASqSums);
    cudaFree(d_BSqSums);
}
