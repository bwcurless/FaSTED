#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <iostream>
#include <vector>

#include "findPairs.cuh"
#include "sumSquared.cuh"
#include "utils.cuh"

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

__device__ uint get_smid(void);

// ---------- Search Parameters ----------
constexpr float epsilon = 0.1;
constexpr int numPoints = 1024 * 16;
constexpr Mma::mmaShape searchShape{numPoints, numPoints, 64 * 64};

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

    // Kind of a hack but we go to NaN if we let it keep incrementing
    int maxFloat = 32768;
    std::vector<half2> h_AValues{};
    // Fill the vector with increasing half-precision values
    // Note that this gets funny > 2048 because of imprecision of half values
    for (int m = 0; m < searchShape.m; m++) {
        for (int k = 0; k < searchShape.k; k += 2) {
            half2 val{};
            val.x = static_cast<half>(min(maxFloat, m * searchShape.k + k));
            val.y = static_cast<half>(min(maxFloat, m * searchShape.k + k + 1));
            h_AValues.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global A", reinterpret_cast<half*>(h_AValues.data()), searchShape.m,
                          searchShape.k);
    }

    std::vector<half2> h_BValues{};
    // Create identity matrix
    for (int row = 0; row < searchShape.n; row++) {
        for (int col = 0; col < searchShape.k; col += 2) {
            half2 val{0, 0};
            if (col == row)
                val.x = 1;
            else if (col + 1 == row)
                val.y = 1;
            h_BValues.push_back(val);
        }
    }

    if (Debug) {
        PrintMatrix<half>("Global B", reinterpret_cast<half*>(h_BValues.data()), searchShape.n,
                          searchShape.k);
    }

    half2 *d_AValues, *d_BValues;
    TransferPointsToGMem(h_AValues, h_BValues, &d_AValues, &d_BValues);

    printf("Running kernel\n");

    // Compute sums of squared dimensions
    using sumSize = float;
    sumSize *d_ASqSums, *d_BSqSums;
    d_ASqSums = SumSqd::ComputeSquaredSums<sumSize>(d_AValues, searchShape.m, searchShape.k);
    d_BSqSums = SumSqd::ComputeSquaredSums<sumSize>(d_BValues, searchShape.n, searchShape.k);

    cudaEventRecord(start, 0);

    BlockTile::FindPairs(BlockTile::FindPairsParams{epsilon, searchShape, d_iterationCount,
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
