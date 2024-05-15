#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>

#include "kernel_join.h"
#include "params.h"

using namespace nvcuda;
using namespace cooperative_groups;

__global__ void printMatrix(double* matrix, unsigned int nbElements) {
    for (unsigned int i = 0; i < nbElements; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j) {
            printf("%f ", matrix[i * COMPUTE_DIM + j]);
        }
        printf("\n");
    }
}

__global__ void printMatrixTranspose(double* matrix, unsigned int size, unsigned int nbElements) {
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        for (unsigned int j = 0; j < nbElements; ++j) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void printMatrixResult(double* matrix, unsigned int size, unsigned int nbElements) {
    for (unsigned int i = 0; i < nbElements; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void convertDataset(INPUT_DATA_TYPE* in, COMPUTE_TYPE* out, unsigned int nbPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nbPoints) {
        for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
            out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)(in[tid * COMPUTE_DIM + i]);
        }
    }
}

__global__ void preComputedSquaredCoordinates(COMPUTE_TYPE* dataset,
                                              ACCUM_TYPE* preComputeCoordinates,
                                              unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

#if ACCUM_PREC == 64
    double accum[4];
    // TODO Why is this unrolled into 4 separate accumulates?
    for (unsigned int i = 0; i < COMPUTE_DIM; i += 4) {
        accum[0] = dataset[tid * COMPUTE_DIM + i] * dataset[tid * COMPUTE_DIM + i];
        accum[1] = dataset[tid * COMPUTE_DIM + i + 1] * dataset[tid * COMPUTE_DIM + i + 1];
        accum[2] = dataset[tid * COMPUTE_DIM + i + 2] * dataset[tid * COMPUTE_DIM + i + 2];
        accum[3] = dataset[tid * COMPUTE_DIM + i + 3] * dataset[tid * COMPUTE_DIM + i + 3];
        preComputeCoordinates[tid * (COMPUTE_DIM / 4) + (i / 4)] =
            accum[0] + accum[1] + accum[2] + accum[3];
    }
#else
    //		float accum[16];
    for (unsigned int i = 0; i < COMPUTE_DIM; i += 16) {
        float accum = 0.0;
#pragma unroll
        for (unsigned int j = 0; j < 16; ++j) {
            accum += __half2float(dataset[tid * COMPUTE_DIM + i + j]) *
                     __half2float(dataset[tid * COMPUTE_DIM + i + j]);
        }
        preComputeCoordinates[tid * (COMPUTE_DIM / 16) + (i / 16)] = accum;
    }
#endif
}

__global__ void preComputedSquaredCoordinatesComplete(COMPUTE_TYPE* dataset,
                                                      ACCUM_TYPE* preComputeCoordinates,
                                                      unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    ACCUM_TYPE accum = 0.0;
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        accum += (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i]) *
                 (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i]);
    }
    preComputeCoordinates[tid] = accum;
}

__global__ void transposeDataset(COMPUTE_TYPE* inputDataset, COMPUTE_TYPE* outputDataset,
                                 unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < COMPUTE_DIM; ++i) {
        outputDataset[tid * COMPUTE_DIM + i] = inputDataset[i * COMPUTE_DIM + tid];
    }
}

__global__ void fillResultMatrix(ACCUM_TYPE* preComputedSquaredCoordinates,
                                 ACCUM_TYPE* resultMatrix, unsigned int nbQueryPoints) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i) {
        resultMatrix[i * nbQueryPoints + tid] = preComputedSquaredCoordinates[tid];
    }
}

__global__ void finishResultMatrix(ACCUM_TYPE* preComputedSquaredCoordinates,
                                   ACCUM_TYPE* resultMatrix, unsigned int nbQueryPoints,
                                   unsigned long long* cnt, ACCUM_TYPE* epsilon) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid) {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i) {
        ACCUM_TYPE finalDistance =
            fabs(resultMatrix[i * nbQueryPoints + tid] + preComputedSquaredCoordinates[i]);

#if ACCUM_PREC == 16
        if (hsqrt(finalDistance) <= (*epsilon))
#else
        if (sqrt(finalDistance) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


// CUDA cores kernel
// Uses 1 thread to compute the distance between 1 query point and all the dataset points
// Uses the standard/textbook Euclidean distance formula
__global__ void distanceCalculationBruteForceCuda(unsigned int* nbQueryPoints,
                                                  COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                  unsigned long long* cnt) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Uses 1 thread per point, so if the thread id is greater than the number of query points, we
    // return
    if ((*nbQueryPoints) <= tid) {
        return;
    }

    // Put the query point in a local array
    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i) {
        point[i] = dataset[tid * COMPUTE_DIM + i];
    }

    // For each dataset point, compute the distance
    for (unsigned int i = 0; i < (*nbQueryPoints); ++i) {
        ACCUM_TYPE accumDistance = 0.0;
        // Loop over all the dimensions
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
            // Standard/textbook Euclidean distance formula
            accumDistance += (ACCUM_TYPE)((point[j] - dataset[i * COMPUTE_DIM + j]) *
                                          (point[j] - dataset[i * COMPUTE_DIM + j]));
        }

#if ACCUM_PREC == 16
        if (hsqrt(accumDistance) <= (*epsilon))
#else
        if (sqrt(accumDistance) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}

// CUDA cores kernel
// Uses 1 thread to compute the distance between 1 query point and all the dataset points
// Uses the extended Euclidean distance formula
__global__ void distanceCalculationBruteForceCudaAlt(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*nbQueryPoints) <= tid) {
        return;
    }

    // Also compute the squared coordinates of the query points
    // since it's being use many times over and over during distance calculations
    COMPUTE_TYPE point[INPUT_DATA_DIM];
    ACCUM_TYPE q2[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i) {
        point[i] = dataset[tid * COMPUTE_DIM + i];
        q2[i] = (ACCUM_TYPE)(point[i]) * (ACCUM_TYPE)(point[i]);
    }

    // Iterate over the dataset points
    for (unsigned int i = 0; i < (*nbQueryPoints); ++i) {
        ACCUM_TYPE accumDistance = 0.0;
        // Iterate over the dimensions
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
            // Extended Euclidean distance formula
            ACCUM_TYPE c2 = (ACCUM_TYPE)(dataset[i * COMPUTE_DIM + j]) *
                            (ACCUM_TYPE)(dataset[i * COMPUTE_DIM + j]);
            accumDistance +=
                (ACCUM_TYPE)((COMPUTE_TYPE)(-2.0) * point[j] * dataset[i * COMPUTE_DIM + j]) +
                q2[j] + c2;
        }

#if ACCUM_PREC == 16
        if (hsqrt(habs(accumDistance)) <= (*epsilon))
#else
        if (sqrt(fabs(accumDistance)) <= (*epsilon))
#endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


// Only compile if using half precision computation (FP16)
#if COMPUTE_PREC == 16

// Tensor cores kernel
// Uses 1 warp to compute the distance between 1 query point and all the dataset points
// Uses the standard/textbook Euclidean distance formula
__global__ void distanceCalculationBruteForceTensorBasic(unsigned int* nbQueryPoints,
                                                         COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                         COMPUTE_TYPE* identityMatrix,
                                                         unsigned long long* cnt) {
    // One query point per warp
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * COMPUTE_DIM];
    // All the intermediate MMA results for each warp
    __shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * 16 * 16];
    __shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * 16 * 16];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    // Since we use 1 warp per query point, the query point is the warp id in the grid
    unsigned int queryPoint = warpIdInGrid;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    // Offset to obtain this thread's warp's results from MMA
    unsigned int sharedArrayResultOffset = warpIdInBlock * 16 * 16;

    if ((*nbQueryPoints) <= queryPoint) {
        return;
    }

    // Statically defined cooperative thread group of 32 threads (1 warp), partitioned
    // from this block of threads
    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
    unsigned int halfWarpId = warp.thread_rank() / 16;
    unsigned int halfWarpThreadId = warp.thread_rank() % 16;

    // Create fragments of the larger matrix we want to multiply
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> matrixAFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> matrixBFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> identityFragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> firstStepAccumulator;
    wmma::fragment<wmma::accumulator, 16, 16, 16, ACCUM_TYPE> secondStepAccumulator;

    // Have each thread in this warp load identityMatrix, starting with that index, and
    // striding by 16 across the leading dimensions, into the identityFragment
    wmma::load_matrix_sync(identityFragment, identityMatrix, 16);

    // Each warp will work together to copy over a single query point to shared memory,
    // no matter the dimension of the point
    for (unsigned int j = 0; j < COMPUTE_DIM; j += WARP_SIZE) {
        // Each thread copies a single value, we stop when we have copied the entire
        // query point over
        if ((j + warp.thread_rank()) < COMPUTE_DIM) {
            sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j + warp.thread_rank()] =
                // Grabbing the query point from dataset as base address
                // Each thread out of 32 is grabbing one element and copying it over
                // j is 0 for all threads for the first iteration, and each thread offsets
                // by it's rank in the warp
                dataset[queryPoint * COMPUTE_DIM + j + warp.thread_rank()];
        }
    }

    // We can calculate the distance between this query point and 16 candidates at once
    // because the matrices are 16x16
    for (unsigned int i = 0; i < (*nbQueryPoints); i += 16) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(16, nbCandidatesLeft);

        wmma::fill_fragment(secondStepAccumulator, 0.0);

        // Can compute up to 16 dimensions at a time because of the size of the matrices
        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16) {
            // Fill A with a part of our query point that this warp is responsible for. Might take
            // Multiple iterations if COMPUTE_DIM is larger than 16 for example.
            // leading dimension of zero because we are duplicating this point across all rows of A
            wmma::load_matrix_sync(matrixAFragment,
                                   sharedArrayQueryPoints + warpIdInBlock * COMPUTE_DIM + n, 0);
            // Fill accumulator with candidate points. Same thing here, this might take more than
            // one iteration to compute the full distance between candidates and query. Stride by
            // COMPUTE_DIM Here because each row is a new candidate point
            wmma::load_matrix_sync(firstStepAccumulator, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM,
                                   wmma::mem_row_major);
            // Broadcast downcast the candidate points to be half precision and negate them all.
            // This is ok to do because although we don't know the ordering of items in a fragment,
            // because it's a broadcast, order doesn't matter.
            // Make negative because we want to subtract firstStepAccumulator from matrix A
            for (int j = 0; j < firstStepAccumulator.num_elements; ++j) {
                firstStepAccumulator.x[j] = (half)(-1.0) * firstStepAccumulator.x[j];
            }

            // Multiply A by the identity matrix, resulting in just A, then subtract candidate
            // points from it. This is the displacement between the two vectors.
            wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment,
                           firstStepAccumulator);
            // Store first step accululator matrix to our shared memory array
            wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset,
                                    firstStepAccumulator, 16, wmma::mem_row_major);

            // Fill A with the distances between query and candidate points
            wmma::load_matrix_sync(matrixAFragment,
                                   sharedArrayResultFirstStep + sharedArrayResultOffset, 16);
            // Fill B with the same, note that since we said B is column major, this will
            // transpose the distance matrix as it sync's. This will let us compute the proper
            // inner product that would yield the square of the distance
            wmma::load_matrix_sync(matrixBFragment,
                                   sharedArrayResultFirstStep + sharedArrayResultOffset, 16);

            // Multiply A by B + previous distances and store back in second accumulator.
            // This is computing the sum of the squares of the distances. It's also generating
            // a lot of wasted computations since we only need N distances (the number of candidate
            // points) but calculate M x N distances
            wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment,
                           secondStepAccumulator);
        }
        // We've completed computing the distance between this query point and 16 candidates

        // Copy the results to shared memory
        wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset,
                                secondStepAccumulator, 16, wmma::mem_row_major);
        // We only get 16 results from the calculation, but we have 32 threads
        if (warp.thread_rank() < 16 && warp.thread_rank() < nbCandidatesLeft) {
            // The actual distances are stored in the diagonals of the matrix. The rest
            // of the values are essentially garbage.
            // Each thread will check one result
            ACCUM_TYPE resultDistance =
                sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * 16 +
                                            warp.thread_rank()];

            // If the distance is less than epsilon, store it out as a match
#if ACCUM_PREC == 16
            if (hsqrt(resultDistance) <= (*epsilon))
#else
            if (sqrt(resultDistance) <= (*epsilon))
#endif
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }
    }
}

// Tensor cores kernel
// Uses 1 warp to compute the distance between 16 query point and all the dataset points
// Uses the extended Euclidean distance formula
__global__ void distanceCalculationBruteForceTensorHalfOpti(
    unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
    unsigned long long* cnt, ACCUM_TYPE* preComputedSquaredCoordinates) {
    // Shared memory arrays
    // Query points
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * 16 * COMPUTE_DIM];
    //    __shared__ half sharedArrayTmp[WARP_PER_BLOCK * 16 * 16];
    //    __shared__ half sharedArrayCHalf[WARP_PER_BLOCK * 16 * 16];
    // Squared coordinates of the query points, there is one sum of squared entries per 16 dims
    // iteration
    __shared__ ACCUM_TYPE sharedArraySquaredQueries[WARP_PER_BLOCK * 16 * (COMPUTE_DIM / 16)];
    // Squared coordinates for the candidate points being computed (up to 16 at a time)
    __shared__ ACCUM_TYPE sharedArraySquaredCandidates[WARP_PER_BLOCK * 16];
    // Temporary array to store the result of the tensor cores
    __shared__ ACCUM_TYPE sharedArrayResultTmp[WARP_PER_BLOCK * 16 * 16];
    // Final result array to accumulate/store the Euclidean distance between the query points and
    // candidate points
    __shared__ ACCUM_TYPE sharedArrayResult[WARP_PER_BLOCK * 16 * 16];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    // The first query point for this warp, out of 16 total
    unsigned int baseQueryPoint = warpIdInGrid * 16;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    if ((*nbQueryPoints) <= baseQueryPoint) {
        return;
    }

    // Offsets for getting to this warp's parts of the shared memory arrays
    unsigned int sharedArrayQueryOffset = warpIdInBlock * 16 * COMPUTE_DIM;
    unsigned int sharedArrayOffset = warpIdInBlock * 16 * 16;
    unsigned int sharedArraySquaredOffset = warpIdInBlock * 16 * (COMPUTE_DIM / 16);

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<16> tile16 = tiled_partition<16>(warp);

    // We normally batch 16 query points per warp, but we may have fewer than 16 in the last warp
    unsigned int nbQueriesBatch =
        (baseQueryPoint + 16 > (*nbQueryPoints)) ? (*nbQueryPoints) - baseQueryPoint : 16;

    // Page the query points in shared memory
    // Uses all 32 threads of the warp, if COMPUTE_DIM is greater than 31
    // i is the query point
    for (unsigned int i = 0; i < nbQueriesBatch; ++i) {
        // j + warp.thread_rank() is a single dimension of i'th the query point
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] =
                    (COMPUTE_TYPE)(-2.0) *
                    dataset[(baseQueryPoint + i) * COMPUTE_DIM + j + warp.thread_rank()];
            }
        }
    }
    // If the warp is assigned fewer than 16 query points (e.g., the very last warp), fill the
    // remaining slots with 0
    for (unsigned int i = nbQueriesBatch; i < 16; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] = (half)0.0;
            }
        }
    }

    // Page the squared coordinates of the query points
    // Only uses 16 threads for simplicity
    if (warp.thread_rank() < 16) {
        for (unsigned int i = 0; i < (COMPUTE_DIM / 16); ++i) {
            if (warp.thread_rank() < nbQueriesBatch) {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 16 + warp.thread_rank()] =
                    // Each thread is copying each query points sum of squares, they are broken up
                    // into COMPUTE_DIM / 16 individual sums though, so larger compute dimensions
                    // would result in more space required.
                    preComputedSquaredCoordinates[(baseQueryPoint + warp.thread_rank()) *
                                                      (COMPUTE_DIM / 16) +
                                                  i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 16 + warp.thread_rank()] =
                    (ACCUM_TYPE)0.0;
            }
        }
    }

    // Iterate over the dataset points
    for (unsigned int i = 0; i < (*nbQueryPoints); i += 16) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(16, nbCandidatesLeft);

        // Set the result array to 0 for the current candidate points
        for (unsigned int j = 0; j < 16; j += 2) {
            sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 +
                              tile16.thread_rank()] = (ACCUM_TYPE)0.0;
        }

        // Iterate over the dimensions
        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16) {
            // Page the squared coordinates of the candidate points
            if (warp.thread_rank() < 16) {
                if ((i + warp.thread_rank()) < (*nbQueryPoints)) {
                    // Each thread page one candidate squared coordinate in, up to 16
                    unsigned int candidateId = i + warp.thread_rank();
                    sharedArraySquaredCandidates[warpIdInBlock * 16 + warp.thread_rank()] =
                        preComputedSquaredCoordinates[candidateId * (COMPUTE_DIM / 16) + (n / 16)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * 16 + warp.thread_rank()] =
                        (ACCUM_TYPE)0.0;
                }
            }

            // Fragments (i.e., matrices) for the MMA operations using the tensor cores
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> matrixQ;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> matrixC;
            wmma::fragment<wmma::accumulator, 16, 16, 16, ACCUM_TYPE> matrixC2;
            wmma::fragment<wmma::accumulator, 16, 16, 16, ACCUM_TYPE> matrixQCC2;

            // Load the query points and candidate points into the fragments
            // Query points are loaded from shared memory
            // Candidate points can be loaded from global memory since the accesses are coalesced
            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n,
                                   COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + (warpIdInBlock * 16), 0,
                                   wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            // Perform the MMA operation (Q x C + C2, where Q is the query points, C the candidate
            // points, and C2 the squared candidate points
            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            // store that intermediary result into the temporary array
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 16,
                                    wmma::mem_row_major);

            // Finish computing the Euclidean distance using CUDA cores, add Q^2, and accumulate
            for (unsigned int j = 0; j < 16; j += 2) {
                // Accumulate the previous result (from tensor cores), the Euclidean distance from
                // previous dimensions, and the squared coordinates of the query points (stored in
                // shared memory)
                unsigned int localId =
                    sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank();
                sharedArrayResult[localId] =
                    sharedArrayResult[localId] + sharedArrayResultTmp[localId] +
                    sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 16) * 16 +
                                              tile16.meta_group_rank() + j];
            }
        }  // for COMPUTE_DIM

        // The Euclidean distance between the query points and the candidate points is now computed
        // Check for each pair if the distance is within epsilon or not
        for (unsigned int j = 0; j < 16; j += 2) {
            if ((j + tile16.meta_group_rank()) < nbQueriesBatch &&
                tile16.thread_rank() < nbCandidatesCurrent) {
                ACCUM_TYPE tmpDistance =
                    abs(sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 +
                                          tile16.thread_rank()]);
#if ACCUM_PREC == 16
                if (hsqrt(tmpDistance) <= (*epsilon))
#else
                if (sqrt(tmpDistance) <= (*epsilon))
#endif
                {
                    unsigned int tmpIdx = atomicAdd(cnt, int(1));
                }
            }
        }
    }  // for nbQueryPoints
}

__global__ void distanceTCFullySummed_16x16x16(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                               ACCUM_TYPE* epsilon, unsigned long long* cnt,
                                               ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCFullySummed<16, 16, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                      preComputedSquaredCoordinates);
}

__global__ void distanceTCFullySummed_8x32x16(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                              ACCUM_TYPE* epsilon, unsigned long long* cnt,
                                              ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCFullySummed<8, 32, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                     preComputedSquaredCoordinates);
}

__global__ void distanceTCFullySummed_32x8x16(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                              ACCUM_TYPE* epsilon, unsigned long long* cnt,
                                              ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCFullySummed<32, 8, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                     preComputedSquaredCoordinates);
}

// Variable Matrix Size Tensor cores kernel
// Uses 1 warp to compute the distance between M query points and N candidate points at a time
// Calculates with K dimensions at a time
// Uses the extended Euclidean distance formula
// Uses the fully summed squared coordinates to speed up computation
template <int Md, int Nd, int Kd>
__device__ void distanceTCFullySummed(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                      float* epsilon, unsigned long long* cnt,
                                      float* preComputedSquaredCoordinates) {
    unsigned long count = 0;
    // Shared memory arrays
    // Query points
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * Md * COMPUTE_DIM];
    // Squared coordinates of the query points, there is one sum of squared entries per query point
    __shared__ float sharedArraySquaredQueries[WARP_PER_BLOCK * Md];
    // Squared coordinates for the candidate points being computed (N at a time)
    __shared__ float sharedArraySquaredCandidates[WARP_PER_BLOCK * Nd];
    // Final result array to accumulate/store the Euclidean distance between the query points and
    // candidate points
    __shared__ float sharedArrayResult[WARP_PER_BLOCK * Md * Nd];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    // The first query point for this warp, out of M total
    unsigned int firstQueryPoint = warpIdInGrid * Md;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    if (firstQueryPoint >= (*nbQueryPoints)) {
        return;
    }

    // Offsets for getting to this warp's parts of the shared memory arrays
    unsigned int sharedArrayQueryOffset = warpIdInBlock * Md * COMPUTE_DIM;
    unsigned int sharedArraySquaredQueriesOffset = warpIdInBlock * Md;
    unsigned int sharedArrayResultOffset = warpIdInBlock * Md * Nd;

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());

    // We normally batch M query points per warp, but we may have fewer than M in the last warp
    unsigned int nbQueriesBatch = min(Md, (*nbQueryPoints) - firstQueryPoint);

    // Page the query points into shared memory
    for (unsigned int i = 0; i < nbQueriesBatch; ++i) {
        // j + warp.thread_rank() is a single dimension of i'th the query point
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] =
                    (COMPUTE_TYPE)(-2.0) *
                    dataset[(firstQueryPoint + i) * COMPUTE_DIM + j + warp.thread_rank()];
            }
        }
    }
    // If the warp is assigned fewer than 16 query points (e.g., the very last warp), fill the
    // remaining slots with 0
    for (unsigned int i = nbQueriesBatch; i < Md; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] = (half)0.0;
            }
        }
    }

    // Page the squared coordinates of the query points
    // Only uses Md threads for simplicity (This works for M <= 32)
    // Each thread copies over one query point's fully summed squared coordinates
    if (warp.thread_rank() < Md) {
        if (warp.thread_rank() < nbQueriesBatch) {
            sharedArraySquaredQueries[sharedArraySquaredQueriesOffset + warp.thread_rank()] =
                preComputedSquaredCoordinates[firstQueryPoint + warp.thread_rank()];
        } else {
            sharedArraySquaredQueries[sharedArraySquaredQueriesOffset + warp.thread_rank()] =
                (float)0.0;
        }
    }

    // Iterate over the dataset points
    for (unsigned int baseCandidate = 0; baseCandidate < (*nbQueryPoints); baseCandidate += Nd) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - baseCandidate;
        unsigned int nbCandidatesCurrent = min(Nd, nbCandidatesLeft);

        // Set the result array to 0 for the current candidate points, for this warp
        // Need all 32 threads in this warp to fill up M x N elements
        for (unsigned int j = warp.thread_rank(); j < Md * Nd; j += WARP_SIZE) {
            sharedArrayResult[sharedArrayResultOffset + j] = (float)0.0;
        }

        __syncthreads();
        // This warp needs to page the squared coordinates of it's candidate points, 1 value for
        // every candidate. This only has to happen once for every set of candidate points
        // Only do this on one warp in the block, no need to repeat it. All warps in the block can
        // reuse it
        if (threadIdx.x < Nd) {
            unsigned int candidateId = baseCandidate + warp.thread_rank();
            if (candidateId < (*nbQueryPoints)) {
                sharedArraySquaredCandidates[warp.thread_rank()] =
                    preComputedSquaredCoordinates[candidateId];
            } else {
                sharedArraySquaredCandidates[warp.thread_rank()] = (float)0.0;
            }
        }
        __syncthreads();

        // Matrix fragment setups...
        // Fragments (i.e., matrices) for the MMA operations using the tensor cores
        wmma::fragment<wmma::matrix_a, Md, Nd, Kd, half, wmma::row_major> matrixQ;
        wmma::fragment<wmma::matrix_b, Md, Nd, Kd, half, wmma::col_major> matrixC;
        // Declare this once as we repeatedly accumulate into it and need it at the end to extrac
        // the results
        wmma::fragment<wmma::accumulator, Md, Nd, Kd, float> matrixQC;
        // Fill initial result matrix with zeros
        wmma::fill_fragment(matrixQC, 0.0);

        // Iterate over the dimensions of the candidate, one matrix worth at a time
        for (unsigned int baseDim = 0; baseDim < COMPUTE_DIM; baseDim += Kd) {
            // Load the query points and candidate points into the fragments
            // Query points are loaded from shared memory
            // Candidate points can be loaded from global memory since the accesses are coalesced
            // Stride by entire dimension of vector since we paged all the query points in
            wmma::load_matrix_sync(
                matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + baseDim, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC, dataset + baseCandidate * COMPUTE_DIM + baseDim,
                                   COMPUTE_DIM);

            // Perform the MMA operation QC_Accum = Q_partial x C_partial + QC_Accum, where Q is the
            // query points, C the candidate points, and QC_Accum is the previous results from lower
            // dimensions being accumulated into.
            wmma::mma_sync(matrixQC, matrixQ, matrixC, matrixQC);

        }  // for COMPUTE_DIM
           // Extract the final accumulated results
        wmma::store_matrix_sync(sharedArrayResult + sharedArrayResultOffset, matrixQC, Nd,
                                wmma::mem_row_major);

        // The Euclidean distance between the query points and the candidate points is almost
        // computed. Finish computation and check for each pair if the distance is within epsilon or
        // not
        for (unsigned int j = warp.thread_rank(); j < Md * Nd; j += WARP_SIZE) {
            unsigned int queryIndex = j / Nd;
            unsigned int candIndex = j % Nd;
            // The last candidate warp, or last query warp might have fewer than the max # of points
            if (queryIndex < nbQueriesBatch && candIndex < nbCandidatesCurrent) {
                // Need to add in C^2 and Q^2 for each value to finalize distance calculation
                // Take the absolute value in case we end up with a tiny negative number, this
                // distance should still be 0
                float tmpDistance =
                    fabs((sharedArrayResult[sharedArrayResultOffset + j] +
                          sharedArraySquaredQueries[sharedArraySquaredQueriesOffset + queryIndex] +
                          sharedArraySquaredCandidates[candIndex]));

                if (sqrt(tmpDistance) <= (*epsilon)) {
                    count++;
                }
            }
        }

    }  // for nbQueryPoints

    // Sum these up once for every thread
    atomicAdd(cnt, count);
}

__global__ void distanceTCShortCircuitable_16x16x16(unsigned int* nbQueryPoints,
                                                    COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                    unsigned long long* cnt,
                                                    ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCShortCircuitable<16, 16, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                           preComputedSquaredCoordinates);
}

__global__ void distanceTCShortCircuitable_8x32x16(unsigned int* nbQueryPoints,
                                                   COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                   unsigned long long* cnt,
                                                   ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCShortCircuitable<8, 32, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                          preComputedSquaredCoordinates);
}

__global__ void distanceTCShortCircuitable_32x8x16(unsigned int* nbQueryPoints,
                                                   COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                   unsigned long long* cnt,
                                                   ACCUM_TYPE* preComputedSquaredCoordinates) {
    distanceTCShortCircuitable<32, 8, 16>(nbQueryPoints, dataset, epsilon, cnt,
                                          preComputedSquaredCoordinates);
}
// Variable Matrix Size Tensor cores kernel
// Uses 1 warp to compute the distance between M query points and N candidate points at a time
// Calculates with K dimensions at a time
// Uses the extended Euclidean distance formula
// Does not use the fully summed squared coordinates to enable short circuiting
template <int Md, int Nd, int Kd>
__device__ void distanceTCShortCircuitable(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                           float* epsilon, unsigned long long* cnt,
                                           float* preComputedSquaredCoordinates) {
    // Shared memory arrays
    // Query points
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * Md * COMPUTE_DIM];
    // Squared coordinates of the query points, there is one sum of squared entries per warp
    // iteration. # of query points * # of iterations to compute
    __shared__ float sharedArraySquaredQueries[WARP_PER_BLOCK * Md * (COMPUTE_DIM / Kd)];
    // Squared coordinates for the candidate points being computed (N at a time)
    __shared__ float sharedArraySquaredCandidates[WARP_PER_BLOCK * Nd];
    // Temporary array to store the result of the tensor cores
    __shared__ float sharedArrayResultTmp[WARP_PER_BLOCK * Md * Nd];
    // Final result array to accumulate/store the Euclidean distance between the query points
    // and candidate points
    __shared__ float sharedArrayResult[WARP_PER_BLOCK * Md * Nd];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    // The first query point for this warp, out of M total
    unsigned int firstQueryPoint = warpIdInGrid * Md;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    if (firstQueryPoint >= (*nbQueryPoints)) {
        return;
    }

    // Offsets for getting to this warp's parts of the shared memory arrays
    unsigned int sharedArrayQueryOffset = warpIdInBlock * Md * COMPUTE_DIM;
    unsigned int sharedArraySquaredQueriesOffset = warpIdInBlock * Md * (COMPUTE_DIM / Kd);
    unsigned int sharedArraySquaredCandidatesOffset = warpIdInBlock * Nd;
    unsigned int sharedArrayResultOffset = warpIdInBlock * Md * Nd;

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<16> tile16 = tiled_partition<16>(warp);

    unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
    // We normally batch M query points per warp, but we may have fewer than M in the last warp
    unsigned int nbQueriesBatch = min(Md, (*nbQueryPoints) - firstQueryPoint);

    // Page the query points into shared memory
    for (unsigned int i = 0; i < nbQueriesBatch; ++i) {
        // j + warp.thread_rank() is a single dimension of i'th the query point
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] =
                    (COMPUTE_TYPE)(-2.0) *
                    dataset[(firstQueryPoint + i) * COMPUTE_DIM + j + warp.thread_rank()];
            }
        }
    }
    // If the warp is assigned fewer than 16 query points (e.g., the very last warp), fill the
    // remaining slots with 0
    for (unsigned int i = nbQueriesBatch; i < Md; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32) {
            if ((j + warp.thread_rank()) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j +
                                       warp.thread_rank()] = (half)0.0;
            }
        }
    }

    // Page the squared coordinates of the query points
    // Only uses Md threads for simplicity (This works for M <= 32)
    // Each thread copies over all of one query point's squared coordinates (iterate on i)
    // There are COMPUTE_DIM / K total squared coordinates
    if (warp.thread_rank() < Md) {
        for (unsigned int i = 0; i < (COMPUTE_DIM / Kd); ++i) {
            if (warp.thread_rank() < nbQueriesBatch) {
                sharedArraySquaredQueries[sharedArraySquaredQueriesOffset + i * Md +
                                          warp.thread_rank()] =
                    // Each thread is copying each query point's sum of squares, they are broken
                    // up into COMPUTE_DIM / Kd individual sums though, so larger compute
                    // dimensions would result in more space required.
                    preComputedSquaredCoordinates[(firstQueryPoint + warp.thread_rank()) *
                                                      (COMPUTE_DIM / Kd) +
                                                  i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredQueriesOffset + i * Md +
                                          warp.thread_rank()] = (float)0.0;
            }
        }
    }

    // Iterate over the dataset points
    for (unsigned int baseCandidate = 0; baseCandidate < (*nbQueryPoints); baseCandidate += Nd) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - baseCandidate;
        unsigned int nbCandidatesCurrent = min(Nd, nbCandidatesLeft);

        // Set the result array to 0 for the current candidate points, for this warp
        // Need all 32 threads in this warp to fill up M x N elements
        for (unsigned int j = warp.thread_rank(); j < Md * Nd; j += WARP_SIZE) {
            sharedArrayResult[sharedArrayResultOffset + j] = (float)0.0;
        }

        // Iterate over the dimensions of the candidate, one matrix worth at a time
        for (unsigned int baseDim = 0; baseDim < COMPUTE_DIM; baseDim += Kd) {
            // This warp needs to page the squared coordinates of it's candidate points, 1 value
            // for every candidate
            if (warp.thread_rank() < Nd) {
                unsigned int candidateId = baseCandidate + warp.thread_rank();
                if (candidateId < (*nbQueryPoints)) {
                    // Each thread page one candidate squared coordinate in
                    sharedArraySquaredCandidates[warpIdInBlock * Nd + warp.thread_rank()] =
                        // There are COMPUTE_DIM / Kd squared points per candidate, and we are
                        // at the baseDim / Kd th point
                        preComputedSquaredCoordinates[candidateId * (COMPUTE_DIM / Kd) +
                                                      (baseDim / Kd)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * Nd + warp.thread_rank()] =
                        (float)0.0;
                }
            }

            // Fragments (i.e., matrices) for the MMA operations using the tensor cores
            wmma::fragment<wmma::matrix_a, Md, Nd, Kd, half, wmma::row_major> matrixQ;
            wmma::fragment<wmma::matrix_b, Md, Nd, Kd, half, wmma::col_major> matrixC;
            wmma::fragment<wmma::accumulator, Md, Nd, Kd, float> matrixC2;
            wmma::fragment<wmma::accumulator, Md, Nd, Kd, float> matrixQCC2;

            // Load the query points and candidate points into the fragments
            // Query points are loaded from shared memory
            // Candidate points can be loaded from global memory since the accesses are
            // coalesced Stride by entire dimension of vector since we paged all the query
            // points in
            wmma::load_matrix_sync(
                matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + baseDim, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC, dataset + baseCandidate * COMPUTE_DIM + baseDim,
                                   COMPUTE_DIM);
            // 0 here for stride because we repeat the squared value across each row
            wmma::load_matrix_sync(
                matrixC2, sharedArraySquaredCandidates + sharedArraySquaredCandidatesOffset, 0,
                wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            // Perform the MMA operation (Q x C + C2, where Q is the query points, C the
            // candidate points, and C2 the squared candidate points
            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            // store that intermediary result into the temporary array
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayResultOffset, matrixQCC2, Nd,
                                    wmma::mem_row_major);

            // Finish computing the Euclidean distance using CUDA cores, add Q^2, and accumulate
            for (unsigned int j = warp.thread_rank(); j < Md * Nd; j += WARP_SIZE) {
                // Accumulate the previous result (from tensor cores), the Euclidean distance
                // from previous dimensions, and the squared coordinates of the query points
                // (stored in shared memory)
                unsigned int localId = sharedArrayResultOffset + j;
                sharedArrayResult[localId] +=
                    sharedArrayResultTmp[localId] +
                    sharedArraySquaredQueries[sharedArraySquaredQueriesOffset +
                                              // Which iteration we are on in the dimensions, *
                                              // Md for the number of query points per mma, +
                                              // the query point we are on (The row of the
                                              // matrix)
                                              (baseDim / Kd) * Md + (j / Nd)];
            }
        }  // for COMPUTE_DIM

        // The Euclidean distance between the query points and the candidate points is now
        // computed Check for each pair if the distance is within epsilon or not
        for (unsigned int j = warp.thread_rank(); j < Md * Nd; j += WARP_SIZE) {
            // The last candidate warp, or last query warp might have fewer than the max # of
            // points
            if (j < nbQueriesBatch * Nd && j % Nd < nbCandidatesCurrent) {
                float tmpDistance = abs(sharedArrayResult[sharedArrayResultOffset + j]);

                if (sqrt(tmpDistance) <= (*epsilon)) {
                    unsigned int tmpIdx = atomicAdd(cnt, int(1));
                }
            }
        }
    }  // for nbQueryPoints
}
#endif

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


// Only compile this kernel if using double precision (FP64)
#if COMPUTE_PREC == 64

__global__ void distanceCalculationBruteForceTensorDoubleOpti(
    unsigned int* nbQueryPoints, double* dataset, double* epsilon, unsigned long long* cnt,
    double* preComputedSquaredCoordinates) {
    __shared__ double sharedArrayQueryPoints[WARP_PER_BLOCK * 8 * COMPUTE_DIM];
    // __shared__ double sharedArrayTmp8x4[WARP_PER_BLOCK * 8 * 4];
    __shared__ double sharedArraySquaredQueries[WARP_PER_BLOCK * 8 * (COMPUTE_DIM / 4)];
    __shared__ double sharedArraySquaredCandidates[WARP_PER_BLOCK * 8];
    __shared__ double sharedArrayResult[WARP_PER_BLOCK * 8 * 8];
    __shared__ double sharedArrayResultTmp[WARP_PER_BLOCK * 8 * 8];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid * 8;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    unsigned int print = 1;

    if ((*nbQueryPoints) <= queryPoint) {
        return;
    }

    unsigned int sharedArrayQueryOffset = warpIdInBlock * 8 * COMPUTE_DIM;
    // unsigned int sharedArray8x4Offset = warpIdInBlock * 8 * 4;
    unsigned int sharedArraySquaredOffset = warpIdInBlock * 8 * (COMPUTE_DIM / 4);
    unsigned int sharedArrayOffset = warpIdInBlock * 8 * 8;

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<8> tile8 = tiled_partition<8>(warp);
    thread_block_tile<4> tile4 = tiled_partition<4>(warp);

    // unsigned int queryPoint = 0;
    // unsigned int nbQueriesBatch = 0;
    // if (0 == warp.thread_rank())
    // {
    //     queryPoint = atomicAdd(&queryPointIdGlobal, int(8));
    // }
    // queryPoint = __shfl_sync(0xffffffff, queryPoint, 0);

    // if ((*nbQueryPoints) < queryPoint)
    // {
    //     return;
    // }

    // if ((queryPoint + 8) > (*nbQueryPoints))
    // {
    //     nbQueriesBatch = (*nbQueryPoints) - queryPoint;
    // } else {
    //     nbQueriesBatch = 8;
    // }

    unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
    unsigned int nbQueriesBatch =
        (queryPoint + 8 > (*nbQueryPoints)) ? (*nbQueryPoints) - queryPoint : 8;

    // Page query points
    if (tile4.meta_group_rank() < nbQueriesBatch) {
        for (unsigned int i = 0; i < COMPUTE_DIM; i += 4) {
            if ((tile4.thread_rank() + i) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset +
                                       tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() +
                                       i] =
                    dataset[(queryPoint + tile4.meta_group_rank()) * COMPUTE_DIM +
                            tile4.thread_rank() + i];
            }
        }
    } else {
        for (unsigned int i = 0; i < COMPUTE_DIM; i += 4) {
            if ((tile4.thread_rank() + i) < COMPUTE_DIM) {
                sharedArrayQueryPoints[sharedArrayQueryOffset +
                                       tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() +
                                       i] = 0.0;
            }
        }
    }

    // if (0 == queryPoint && 0 == warp.thread_rank())
    // {
    //     printf("\nQuery points: \n");
    //     for (unsigned int i = 0; i < 8; ++i)
    //     {
    //         printf("Query %d: ", i);
    //         for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
    //         {
    //             printf("%f, ", sharedArrayQueryPoints[sharedArrayQueryOffset + i *
    //             COMPUTE_DIM + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    if (warp.thread_rank() < 8) {
        for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i) {
            if (warp.thread_rank() < nbQueriesBatch) {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] =
                    preComputedSquaredCoordinates[(queryPoint + warp.thread_rank()) *
                                                      (COMPUTE_DIM / 4) +
                                                  i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] =
                    0.0;
            }
        }
    }

    // if (0 == queryPoint && 0 == warp.thread_rank() && 1 == print)
    // {
    //     printf("\nSquared queries (Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7): \n");
    //     for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i)
    //     {
    //         for (unsigned int j = 0; j < 8; ++j)
    //         {
    //             printf("%f, ", sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 +
    //             j]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (unsigned int i = 0; i < (*nbQueryPoints); i += 8) {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(8, nbCandidatesLeft);

        sharedArrayResult[sharedArrayOffset + warp.thread_rank()] = 0.0;
        sharedArrayResult[sharedArrayOffset + warp.thread_rank() + 32] = 0.0;

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 4) {
            // if ((i + tile4.meta_group_rank()) < (*nbQueryPoints))
            // {
            //     sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 +
            //     tile4.thread_rank()] =
            //             dataset[(i + tile4.meta_group_rank()) * COMPUTE_DIM + n +
            //             tile4.thread_rank()];
            //     if (0 == tile4.thread_rank())
            //     {
            //         sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()]
            //         =
            //                 preComputedSquaredCoordinates[(i + tile4.meta_group_rank()) *
            //                 (COMPUTE_DIM / 4) + (n / 4)];
            //     }
            // } else {
            //     sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 +
            //     tile4.thread_rank()] = 0.0; if (0 == tile4.thread_rank())
            //     {
            //         sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()]
            //         = 0.0;
            //     }
            // }

            if (warp.thread_rank() < 8) {
                if (warp.thread_rank() < nbCandidatesCurrent) {
                    sharedArraySquaredCandidates[warpIdInBlock * 8 + warp.thread_rank()] =
                        preComputedSquaredCoordinates[(i + warp.thread_rank()) * (COMPUTE_DIM / 4) +
                                                      (n / 4)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * 8 + warp.thread_rank()] = 0.0;
                }
            }

            // if (0 == queryPoint && 0 == warp.thread_rank() && 1 == print)
            // {
            //     printf("\nCandidate points: \n");
            //     for (unsigned int k = 0; k < 8; ++k)
            //     {
            //         printf("Candidate %d: ", k);
            //         for (unsigned int l = 0; l < 4; ++l)
            //         {
            //             printf("%f, ", dataset[(i + k) * COMPUTE_DIM + l]);
            //         }
            //         printf("\n");
            //     }

            //     printf("\nSquared candidate points (C0, C1, C2, C3, C4, C5, C6, C7): \n");
            //     for (unsigned int k = 0; k < 8; ++k)
            //     {
            //         printf("%f, ", sharedArraySquaredCandidates[warpIdInBlock * 8 + k]);
            //     }
            //     printf("\n");
            // }

            wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> matrixQ;
            wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> matrixC;
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixC2;
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixQCC2;

            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n,
                                   COMPUTE_DIM);
            // wmma::load_matrix_sync(matrixC, sharedArrayTmp8x4 + sharedArray8x4Offset, 4);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + warpIdInBlock * 8, 0,
                                   wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            for (unsigned int k = 0; k < matrixQ.num_elements; ++k) {
                matrixQ.x[k] *= (-2.0);
            }

            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 8,
                                    wmma::mem_row_major);

            // if (0 == queryPoint && 0 == warp.thread_rank() && 1 == print)
            // {
            //     printf("\n-2QC + C^2: \n");
            //     for (unsigned int k = 0; k < 8; ++k)
            //     {
            //         printf("Query %d: ", k);
            //         for (unsigned int l = 0; l < 8; ++l)
            //         {
            //             printf("%f, ", sharedArrayResultTmp[sharedArrayOffset + k * 8 + l]);
            //         }
            //         printf("\n");
            //     }
            // }

            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                              tile8.thread_rank()] =
                sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                                  tile8.thread_rank()] +
                sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                                     tile8.thread_rank()] +
                sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 +
                                          tile8.meta_group_rank()];
            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                              tile8.thread_rank() + 32] =
                sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                                  tile8.thread_rank() + 32] +
                sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                                     tile8.thread_rank() + 32] +
                sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 +
                                          tile8.meta_group_rank() + 4];

            // if (0 == queryPoint && 0 == warp.thread_rank() && 1 == print)
            // {
            //     printf("\nResult: \n");
            //     for (unsigned int k = 0; k < 8; ++k)
            //     {
            //         printf("Query %d: ", k);
            //         for (unsigned int l = 0; l < 8; ++l)
            //         {
            //             printf("%f, ", sharedArrayResult[sharedArrayOffset + k * 8 + l]);
            //         }
            //         printf("\n");
            //     }
            // }

            print = 0;
        }  // for COMPUTE_DIM

        if (tile8.meta_group_rank() < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent) {
            double tmpDistance =
                fabs(sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 +
                                       tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon)) {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }
        if ((tile8.meta_group_rank() + 4) < nbQueriesBatch &&
            tile8.thread_rank() < nbCandidatesCurrent) {
            double tmpDistance =
                fabs(sharedArrayResult[sharedArrayOffset + 32 + tile8.meta_group_rank() * 8 +
                                       tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon)) {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }

    }  // for nbQueryPoints
}

#endif  // COMPUTE_PREC == 64
