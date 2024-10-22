#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "DataLoader/PointListBuilder.hpp"
#include "findPairs.cuh"
#include "sumSquared.cuh"
#include "utils.cuh"

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

/** Create a set of points with monatomically increasing values. Increments by 1 for every point.
 * Note that half values can only count up to about 64000, so the max value is
 * capped at 32768.
 *
 * \param values The vector to push the values onto.
 * \param numPoints How many points to create.
 * \param numDimensions How many dimensions per point.
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
 * \param numPoints How many points to create.
 * \param numDimensions How many dimensions per point.
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
void TransferPointsToGMem(const std::vector<Precision>& h_Query,
                          const std::vector<Precision>& h_Candidate, T** d_Query, T** d_Candidate) {
    size_t querySize = h_Query.size() * sizeof(h_Query[0]);
    size_t candidateSize = h_Candidate.size() * sizeof(h_Candidate[0]);

    if (Debug) {
        printf("Query vector has %lu elements\n", h_Query.size());
        printf("Candidate vector has %lu elements\n", h_Candidate.size());
        printf("Allocating %lu bytes for query points\n", querySize);
        printf("Allocating %lu bytes for candidate points\n", candidateSize);
    }

    cudaMalloc(d_Query, querySize);
    cudaMalloc(d_Candidate, candidateSize);

    cudaMemcpy(*d_Query, h_Query.data(), querySize, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_Candidate, h_Candidate.data(), candidateSize, cudaMemcpyHostToDevice);
}

double parseDouble(std::string str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("Invalid argument: unable to parse double");
    } catch (const std::out_of_range& e) {
        throw std::out_of_range("Out of range: the number is too large or too small for a double");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <epsilon>" << std::endl;
        return 1;  // Exit with error if the number of arguments is incorrect
    }

    // Set the global locale to the default locale (which should use commas for thousands separator
    // in the US)
    std::cout.imbue(std::locale("en_US.UTF-8"));

    std::string filename = argv[1];       // Get the filename from the command-line argument
    std::string epsilonString = argv[2];  // Get epsilon from the command-line argument
    double epsilon = parseDouble(epsilonString);

    half2 *d_AValues, *d_BValues;
    // Attempt to build the PointList using the provided filename
    Mma::mmaShape bDims = BlockTile::GetBlockTileDims();
    Points::PointList<half_float::half> pointList;
    if (Debug) {
        pointList =
            Points::PointListBuilder<half_float::half>().withMaxPoints(2000).buildFromAsciiFile(
                filename, ',', bDims.k, bDims.m);
    } else {
        pointList = Points::PointListBuilder<half_float::half>().buildFromAsciiFile(
            filename, ',', 16 * bDims.k, bDims.m);
    }

    Mma::mmaShape searchShape{pointList.getNumPoints(), pointList.getNumPoints(),
                              pointList.getDimensions()};
    Mma::mmaShape actualSearchShape{pointList.getActualNumPoints(), pointList.getActualNumPoints(),
                                    pointList.getActualDimensions()};

    std::cout << "Padded Search Dimensions:" << std::endl;
    std::cout << "M: " << searchShape.m << std::endl;
    std::cout << "N: " << searchShape.n << std::endl;
    std::cout << "K: " << searchShape.k << std::endl;

    std::cout << "Actual Search Dimensions:" << std::endl;
    std::cout << "M: " << actualSearchShape.m << std::endl;
    std::cout << "N: " << actualSearchShape.n << std::endl;
    std::cout << "K: " << actualSearchShape.k << std::endl;

    if (Debug) {
        PrintMatrix<half>("Dataset A", reinterpret_cast<half*>(pointList.values.data()),
                          searchShape.m, searchShape.k);
    }

    TransferPointsToGMem(pointList.values, pointList.values, &d_AValues, &d_BValues);
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Keep track of how many pairs we found that are within epsilon of each other
    unsigned long long* d_numPairs;
    unsigned long long h_numPairs = 0;
    cudaMalloc(&d_numPairs, sizeof(unsigned long long));
    cudaMemcpy(d_numPairs, &h_numPairs, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    printf("Running kernel\n");

    cudaEventRecord(start, 0);

    // Compute sums of squared dimensions
    using sumSize = float;
    sumSize *d_ASqSums, *d_BSqSums;
    d_ASqSums = SumSqd::ComputeSquaredSums<sumSize>(d_AValues, searchShape.m, searchShape.k);
    d_BSqSums = SumSqd::ComputeSquaredSums<sumSize>(d_BValues, searchShape.n, searchShape.k);

    float epsilonSquared = epsilon * epsilon;
    auto params =
        BlockTile::FindPairsParams{epsilonSquared, searchShape, actualSearchShape, d_numPairs,
                                   d_AValues,      d_BValues,   d_ASqSums,         d_BSqSums};
    BlockTile::FindPairs(params);

    gpuErrchk(cudaEventRecord(stop, 0));

    gpuErrchk(cudaEventSynchronize(stop));

    cudaMemcpy(&h_numPairs, d_numPairs, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    std::cout << "Number of total pairs: " << h_numPairs << std::endl;

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
    cudaFree(d_numPairs);
    cudaFree(d_AValues);
    cudaFree(d_BValues);
    cudaFree(d_ASqSums);
    cudaFree(d_BSqSums);
}
