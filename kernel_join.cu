#include <math.h>
#include <stdio.h>

#include "kernel_join.h"
#include "params.h"

#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace cooperative_groups;


__global__ void printMatrix(double* matrix, unsigned int nbElements)
{
    for (unsigned int i = 0; i < nbElements; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
        {
            printf("%f ", matrix[i * COMPUTE_DIM + j]);
        }
        printf("\n");
    }
}

__global__ void printMatrixTranspose(double* matrix, unsigned int size, unsigned int nbElements)
{
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i)
    {
        for (unsigned int j = 0; j < nbElements; ++j)
        {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


__global__ void printMatrixResult(double* matrix, unsigned int size, unsigned int nbElements)
{
    for (unsigned int i = 0; i < nbElements; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
        {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}


__global__ void convertDataset(
    INPUT_DATA_TYPE* in,
    COMPUTE_TYPE* out,
    unsigned int nbPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nbPoints)
    {
        for (unsigned int i = 0; i < COMPUTE_DIM; ++i)
        {
            out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)(in[tid * COMPUTE_DIM + i]);
        }
    }
}


__global__ void preComputedSquaredCoordinates(
        COMPUTE_TYPE* dataset,
        ACCUM_TYPE* preComputeCoordinates,
        unsigned int nbQueryPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid)
    {
        return;
    }

    #if ACCUM_PREC == 64
        double accum[4];
        for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
        {
            accum[0] = dataset[tid * COMPUTE_DIM + i] * dataset[tid * COMPUTE_DIM + i];
            accum[1] = dataset[tid * COMPUTE_DIM + i + 1] * dataset[tid * COMPUTE_DIM + i + 1];
            accum[2] = dataset[tid * COMPUTE_DIM + i + 2] * dataset[tid * COMPUTE_DIM + i + 2];
            accum[3] = dataset[tid * COMPUTE_DIM + i + 3] * dataset[tid * COMPUTE_DIM + i + 3];
            preComputeCoordinates[tid * (COMPUTE_DIM / 4) + (i / 4)] = accum[0] + accum[1] + accum[2] + accum[3];
        }
    #else
    //		float accum[16];
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 16)
		{
            float accum = 0.0;
            #pragma unroll
            for (unsigned int j = 0; j < 16; ++j)
            {
                accum += __half2float(dataset[tid * COMPUTE_DIM + i + j]) * __half2float(dataset[tid * COMPUTE_DIM + i + j]);
            }
            preComputeCoordinates[tid * (COMPUTE_DIM / 16) + (i / 16)] = accum;
		}
    #endif
}


__global__ void preComputedSquaredCoordinatesComplete(
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* preComputeCoordinates,
    unsigned int nbQueryPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid)
    {
        return;
    }

    ACCUM_TYPE accum = 0.0;
    for (unsigned int i = 0; i < COMPUTE_DIM; ++i)
    {
        accum += (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i]) * (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i]);
    }
    preComputeCoordinates[tid] = accum;
}


__global__ void transposeDataset(
    COMPUTE_TYPE* inputDataset,
    COMPUTE_TYPE* outputDataset,
    unsigned int nbQueryPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (nbQueryPoints <= tid)
    {
        return;
    }

    for (unsigned int i = 0; i < COMPUTE_DIM; ++i)
    {
        outputDataset[tid * COMPUTE_DIM + i] = inputDataset[i * COMPUTE_DIM + tid];
    }
}


__global__ void fillResultMatrix(
    ACCUM_TYPE* preComputedSquaredCoordinates,
    ACCUM_TYPE* resultMatrix,
    unsigned int nbQueryPoints)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid)
    {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i)
    {
        resultMatrix[i * nbQueryPoints + tid] = preComputedSquaredCoordinates[tid];
    }
}


__global__ void finishResultMatrix(
    ACCUM_TYPE* preComputedSquaredCoordinates,
    ACCUM_TYPE* resultMatrix,
    unsigned int nbQueryPoints,
    unsigned long long* cnt,
    ACCUM_TYPE* epsilon)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbQueryPoints <= tid)
    {
        return;
    }

    for (unsigned int i = 0; i < nbQueryPoints; ++i)
    {
        ACCUM_TYPE finalDistance = fabs(resultMatrix[i * nbQueryPoints + tid] + preComputedSquaredCoordinates[i]);

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
__global__ void distanceCalculationBruteForceCuda(
    unsigned int* nbQueryPoints,
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* epsilon,
    unsigned long long* cnt)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Uses 1 thread per point, so if the thread id is greater than the number of query points, we return
    if ((*nbQueryPoints) <= tid)
    {
        return;
    }

    // Put the query point in a local array
    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i)
    {
        point[i] = dataset[tid * COMPUTE_DIM + i];
    }

    // For each dataset point, compute the distance
    for (unsigned int i = 0; i < (*nbQueryPoints); ++i)
    {
        ACCUM_TYPE accumDistance = 0.0;
        // Loop over all the dimensions
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j)
        {
            // Standard/textbook Euclidean distance formula
            accumDistance += (ACCUM_TYPE)((point[j] - dataset[i * COMPUTE_DIM + j]) * (point[j] - dataset[i * COMPUTE_DIM + j]));
        }

        #if ACCUM_PREC == 16
        if(hsqrt(accumDistance) <= (*epsilon))
        #else
        if(sqrt(accumDistance) <= (*epsilon))
        #endif
        {
            unsigned int idx = atomicAdd(cnt, int(1));
        }
    }
}


// CUDA cores kernel
// Uses 1 thread to compute the distance between 1 query point and all the dataset points
// Uses the extended Euclidean distance formula
__global__ void distanceCalculationBruteForceCudaAlt(
        unsigned int* nbQueryPoints,
        COMPUTE_TYPE* dataset,
        ACCUM_TYPE* epsilon,
        unsigned long long* cnt)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*nbQueryPoints) <= tid)
    {
        return;
    }

    // Also compute the squared coordinates of the query points
    // since it's being use many times over and over during distance calculations
    COMPUTE_TYPE point[INPUT_DATA_DIM];
    ACCUM_TYPE q2[INPUT_DATA_DIM];
    for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i)
    {
        point[i] = dataset[tid * COMPUTE_DIM + i];
        q2[i] = (ACCUM_TYPE)(point[i]) * (ACCUM_TYPE)(point[i]);
    }

    // Iterate over the dataset points
    for (unsigned int i = 0; i < (*nbQueryPoints); ++i)
    {
        ACCUM_TYPE accumDistance = 0.0;
        // Iterate over the dimensions
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j)
        {
            // Extended Euclidean distance formula
            ACCUM_TYPE c2 = (ACCUM_TYPE)(dataset[i * COMPUTE_DIM + j]) * (ACCUM_TYPE)(dataset[i * COMPUTE_DIM + j]);
            accumDistance += (ACCUM_TYPE)((COMPUTE_TYPE)(-2.0) * point[j] * dataset[i * COMPUTE_DIM + j]) + q2[j] + c2;
        }

        #if ACCUM_PREC == 16
        if(hsqrt(habs(accumDistance)) <= (*epsilon))
        #else
        if(sqrt(fabs(accumDistance)) <= (*epsilon))
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
__global__ void distanceCalculationBruteForceTensorBasic(
    unsigned int* nbQueryPoints,
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* epsilon,
    COMPUTE_TYPE* identityMatrix,
    unsigned long long* cnt)
{
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * COMPUTE_DIM];
    __shared__ half sharedArrayResultFirstStep[WARP_PER_BLOCK * 16 * 16];
    __shared__ ACCUM_TYPE sharedArrayResultSecondStep[WARP_PER_BLOCK * 16 * 16];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    unsigned int sharedArrayResultOffset = warpIdInBlock * 16 * 16;

    if ((*nbQueryPoints) <= queryPoint)
    {
        return;
    }

    thread_block_tile<WARP_SIZE> warp = tiled_partition<WARP_SIZE>(this_thread_block());
    unsigned int halfWarpId = warp.thread_rank() / 16;
    unsigned int halfWarpThreadId = warp.thread_rank() % 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> matrixAFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> matrixBFragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> identityFragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> firstStepAccumulator;
    wmma::fragment<wmma::accumulator, 16, 16, 16, ACCUM_TYPE> secondStepAccumulator;

    wmma::load_matrix_sync(identityFragment, identityMatrix, 16);

    for (unsigned int j = 0; j < COMPUTE_DIM; j += WARP_SIZE)
    {
        if ((j + warp.thread_rank()) < COMPUTE_DIM)
        {
            sharedArrayQueryPoints[warpIdInBlock * COMPUTE_DIM + j + warp.thread_rank()] = dataset[queryPoint * COMPUTE_DIM + j + warp.thread_rank()];
        }
    }

    for (unsigned int i = 0; i < (*nbQueryPoints); i += 16)
    {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(16, nbCandidatesLeft);

        wmma::fill_fragment(secondStepAccumulator, 0.0);

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16)
        {
            wmma::load_matrix_sync(matrixAFragment, sharedArrayQueryPoints + warpIdInBlock * COMPUTE_DIM + n, 0);
            wmma::load_matrix_sync(firstStepAccumulator, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM, wmma::mem_row_major);
            for (int j = 0; j < firstStepAccumulator.num_elements; ++j)
            {
                firstStepAccumulator.x[j] = (half)(-1.0) * firstStepAccumulator.x[j];
            }

            wmma::mma_sync(firstStepAccumulator, matrixAFragment, identityFragment, firstStepAccumulator);
            wmma::store_matrix_sync(sharedArrayResultFirstStep + sharedArrayResultOffset, firstStepAccumulator, 16, wmma::mem_row_major);

            wmma::load_matrix_sync(matrixAFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, 16);
            wmma::load_matrix_sync(matrixBFragment, sharedArrayResultFirstStep + sharedArrayResultOffset, 16);

            wmma::mma_sync(secondStepAccumulator, matrixAFragment, matrixBFragment, secondStepAccumulator);
        }

        wmma::store_matrix_sync(sharedArrayResultSecondStep + sharedArrayResultOffset, secondStepAccumulator, 16, wmma::mem_row_major);
        if (warp.thread_rank() < 16 && warp.thread_rank() < nbCandidatesLeft)
        {
            ACCUM_TYPE resultDistance = sharedArrayResultSecondStep[sharedArrayResultOffset + warp.thread_rank() * 16 + warp.thread_rank()];

            #if ACCUM_PREC == 16
            if(hsqrt(__habsresultDistance) <= (*epsilon))
            #else
            if(sqrt(resultDistance) <= (*epsilon))
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
// TODO: Copy and paste this kernel, and make the necessary change to use 32x8x16 and 8x32x16 matrix multiplication instead of 16x16x16
// TODO: Overall, for every '16' you see in this code, ask yourself if it should be replaced by '8' or '32'
__global__ void distanceCalculationBruteForceTensorHalfOpti(
    unsigned int* nbQueryPoints,
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* epsilon,
    unsigned long long* cnt,
    ACCUM_TYPE* preComputedSquaredCoordinates)
{
    // Shared memory arrays
    // Query points
    __shared__ half sharedArrayQueryPoints[WARP_PER_BLOCK * 16 * COMPUTE_DIM];
//    __shared__ half sharedArrayTmp[WARP_PER_BLOCK * 16 * 16];
//    __shared__ half sharedArrayCHalf[WARP_PER_BLOCK * 16 * 16];
    // Squared coordinates of the query points
    __shared__ ACCUM_TYPE sharedArraySquaredQueries[WARP_PER_BLOCK * 16 * (COMPUTE_DIM / 16)];
    // Squared coordinates for the candidate points being computed (up to 16 at a time)
    __shared__ ACCUM_TYPE sharedArraySquaredCandidates[WARP_PER_BLOCK * 16];
    // Temporary array to store the result of the tensor cores
    __shared__ ACCUM_TYPE sharedArrayResultTmp[WARP_PER_BLOCK * 16 * 16];
    // Final result array to accumulate/store the Euclidean distance between the query points and candidate points
    __shared__ ACCUM_TYPE sharedArrayResult[WARP_PER_BLOCK * 16 * 16];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpIdInGrid = tid / WARP_SIZE;
    unsigned int queryPoint = warpIdInGrid * 16;
    unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

    if ((*nbQueryPoints) <= queryPoint)
    {
        return;
    }

    unsigned int sharedArrayQueryOffset = warpIdInBlock * 16 * COMPUTE_DIM;
    unsigned int sharedArrayOffset = warpIdInBlock * 16 * 16;
    unsigned int sharedArraySquaredOffset = warpIdInBlock * 16 * (COMPUTE_DIM / 16);

    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    thread_block_tile<16> tile16 = tiled_partition<16>(warp);

    unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
    unsigned int nbQueriesBatch = (queryPoint + 16 > (*nbQueryPoints)) ? (*nbQueryPoints) - queryPoint : 16;

    // Page the query points in shared memory
    // Uses all 32 threads ot the warp
    for (unsigned int i = 0; i < nbQueriesBatch; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32)
        {
            if ((j + warp.thread_rank()) < COMPUTE_DIM)
            {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j + warp.thread_rank()] =
                        (COMPUTE_TYPE)(-2.0) * dataset[(queryPoint + i) * COMPUTE_DIM + j + warp.thread_rank()];
            }
        }
    }
    // If the warp is assigned fewer than 16 query points (e.g., the very last warp), fill the remaining slots with 0
    for (unsigned int i = nbQueriesBatch; i < 16; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; j += 32)
        {
            if ((j + warp.thread_rank()) < COMPUTE_DIM)
            {
                sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j + warp.thread_rank()] = (half)0.0;
            }
        }
    }

    // Page the squared coordinates of the query points
    // Only uses 16 threads for simplicity
    if (warp.thread_rank() < 16)
    {
        for (unsigned int i = 0; i < (COMPUTE_DIM / 16); ++i)
        {
            if (warp.thread_rank() < nbQueriesBatch)
            {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 16 + warp.thread_rank()] =
                        preComputedSquaredCoordinates[(queryPoint + warp.thread_rank()) * (COMPUTE_DIM / 16) + i];
            } else {
                sharedArraySquaredQueries[sharedArraySquaredOffset + i * 16 + warp.thread_rank()] = (ACCUM_TYPE)0.0;
            }
        }
    }

    // Iterate over the dataset points
    for (unsigned int i = 0; i < (*nbQueryPoints); i += 16)
    {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(16, nbCandidatesLeft);

        // Set the result array to 0 for the current candidate points
        for (unsigned int j = 0; j < 16; j += 2)
        {
            sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank()] = (ACCUM_TYPE) 0.0;
        }

        // Iterate over the dimensions
        for (unsigned int n = 0; n < COMPUTE_DIM; n += 16)
        {
            // Page the squared coordinates of the candidate points
            if (warp.thread_rank() < 16)
            {
                if ((i + warp.thread_rank()) < (*nbQueryPoints))
                {
                    unsigned int candidateId = i + warp.thread_rank();
                    sharedArraySquaredCandidates[warpIdInBlock * 16 + warp.thread_rank()] =
                            preComputedSquaredCoordinates[candidateId * (COMPUTE_DIM / 16) + (n / 16)];
                } else {
                    sharedArraySquaredCandidates[warpIdInBlock * 16 + warp.thread_rank()] = (ACCUM_TYPE)0.0;
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
            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + (warpIdInBlock * 16), 0, wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            // Perform the MMA operation (Q x C + C2, where Q is the query points, C the candidate points, and C2 the squared candidate points
            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            // store that intermediary result into the temporary array
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 16, wmma::mem_row_major);

            // Finish computing the Euclidean distance using CUDA cores
            for (unsigned int j = 0; j < 16; j += 2)
            {
                // Accumulate the previous result (from tensor cores), the Euclidean distance from previous dimensions,
                // and the squared coordinates of the query points (stored in shared memory)
                unsigned int localId = sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank();
                sharedArrayResult[localId] = sharedArrayResult[localId]
                                            + sharedArrayResultTmp[localId]
                                            + sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 16) * 16 + tile16.meta_group_rank() + j];
            }
        } // for COMPUTE_DIM

        // The Euclidean distance between the query points and the candidate points is now computed
        // Check for each pair if the distance is within epsilon or not
        for (unsigned int j = 0; j < 16; j += 2)
        {
            if ((j + tile16.meta_group_rank()) < nbQueriesBatch && tile16.thread_rank() < nbCandidatesCurrent)
            {
                ACCUM_TYPE tmpDistance = abs(sharedArrayResult[sharedArrayOffset + (j + tile16.meta_group_rank()) * 16 + tile16.thread_rank()]);
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
    } // for nbQueryPoints
}

#endif


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


// Only compile this kernel if using double precision (FP64)
#if COMPUTE_PREC == 64

__global__ void distanceCalculationBruteForceTensorDoubleOpti(
    unsigned int* nbQueryPoints,
    double* dataset,
    double* epsilon,
    unsigned long long* cnt,
    double* preComputedSquaredCoordinates)
{
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

    if ((*nbQueryPoints) <= queryPoint)
    {
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
	unsigned int nbQueriesBatch = (queryPoint + 8 > (*nbQueryPoints)) ? (*nbQueryPoints) - queryPoint : 8;

	// Page query points
	if (tile4.meta_group_rank() < nbQueriesBatch)
	{
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
		{
			if ((tile4.thread_rank() + i) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] =
					dataset[(queryPoint + tile4.meta_group_rank()) * COMPUTE_DIM + tile4.thread_rank() + i];
			}
		}
	} else {
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
		{
			if ((tile4.thread_rank() + i) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] = 0.0;
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
    //             printf("%f, ", sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    if (warp.thread_rank() < 8)
	{
		for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i)
		{
			if (warp.thread_rank() < nbQueriesBatch)
			{
				sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] =
					preComputedSquaredCoordinates[(queryPoint + warp.thread_rank()) * (COMPUTE_DIM / 4) + i];
			} else {
				sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] = 0.0;
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
    //             printf("%f, ", sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (unsigned int i = 0; i < (*nbQueryPoints); i += 8)
    {
        unsigned int nbCandidatesLeft = (*nbQueryPoints) - i;
        unsigned int nbCandidatesCurrent = min(8, nbCandidatesLeft);

        sharedArrayResult[sharedArrayOffset + warp.thread_rank()] = 0.0;
        sharedArrayResult[sharedArrayOffset + warp.thread_rank() + 32] = 0.0;

        for (unsigned int n = 0; n < COMPUTE_DIM; n += 4)
        {
            // if ((i + tile4.meta_group_rank()) < (*nbQueryPoints))
            // {
            //     sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 + tile4.thread_rank()] =
            //             dataset[(i + tile4.meta_group_rank()) * COMPUTE_DIM + n + tile4.thread_rank()];
            //     if (0 == tile4.thread_rank())
            //     {
            //         sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()] = 
            //                 preComputedSquaredCoordinates[(i + tile4.meta_group_rank()) * (COMPUTE_DIM / 4) + (n / 4)];
            //     }
            // } else {
            //     sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 + tile4.thread_rank()] = 0.0;
            //     if (0 == tile4.thread_rank())
            //     {
            //         sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()] = 0.0;
            //     }
            // }

            if (warp.thread_rank() < 8)
            {
                if (warp.thread_rank() < nbCandidatesCurrent)
                {
                    sharedArraySquaredCandidates[warpIdInBlock * 8 + warp.thread_rank()] =
                        preComputedSquaredCoordinates[(i + warp.thread_rank()) * (COMPUTE_DIM / 4) + (n / 4)];
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

            wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n, COMPUTE_DIM);
            // wmma::load_matrix_sync(matrixC, sharedArrayTmp8x4 + sharedArray8x4Offset, 4);
            wmma::load_matrix_sync(matrixC, dataset + i * COMPUTE_DIM + n, COMPUTE_DIM);
            wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + warpIdInBlock * 8, 0, wmma::mem_row_major);
            wmma::fill_fragment(matrixQCC2, 0.0);

            for (unsigned int k = 0; k < matrixQ.num_elements; ++k)
            {
                matrixQ.x[k] *= (-2.0);
            }

            wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
            wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 8, wmma::mem_row_major);

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

            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()] =
                    sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
                    + sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
                    + sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank()];
            sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32] =
                    sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
                    + sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
                    + sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank() + 4];

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
        } // for COMPUTE_DIM

        if (tile8.meta_group_rank() < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
        {
            double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon))
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }
        if ((tile8.meta_group_rank() + 4) < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
        {
            double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + 32 + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
            if (sqrt(tmpDistance) <= (*epsilon))
            {
                unsigned int tmpIdx = atomicAdd(cnt, int(1));
            }
        }

    } // for nbQueryPoints
}

#endif // COMPUTE_PREC == 64