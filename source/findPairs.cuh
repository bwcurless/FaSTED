/******************************************************************************
1* File:             findPairs.cuh
 * * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      Responsible for finding all points that are within epsilon of each other, given
a dataset already loaded into host memory.
 *****************************************************************************/

#ifndef FINDPAIRS_CUH_KP5RAZNA
#define FINDPAIRS_CUH_KP5RAZNA

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <omp.h>

#include <cuda/pipeline>

#include "DataLoader/PointList.hpp"
#include "matrix.cuh"
#include "pair.cuh"
#include "ptxMma.cuh"
#include "rasterizer/rasterizer.hpp"
#include "sumSquared.cuh"
#include "utils.cuh"
#include "warpMma.cuh"

namespace SimSearch {

/** Release the global memory allocated to hold the input data for the A and B arrays.
 *
 */
__host__ void releaseGlobalMemory();

struct Results {
    double TFLOPS;                  // The estimated teraflops achieved
    unsigned long long pairsFound;  // How many pairs were found
    unsigned long long
        pairsStored;  // How many pairs were actually saved. (Could run out of memory)
    Mma::mmaShape inputProblemShape;   // The input shape of the problem, m x n x k
    Mma::mmaShape paddedProblemShape;  // The actual shape of the problem, m x n x k (includes
                                       // padding of 0's)
};

using Coordinate = matrix::Coordinate;

// Allocate the space for the input data on the device so that we can reuse it in-between
// invocations while fine tuning epsilon.
half2 *d_AValues, *d_BValues = nullptr;

// Block Parameters
constexpr int numWarpCols = 2;
constexpr int numWarpRows = 2;
constexpr int numWarps = numWarpCols * numWarpRows;
constexpr int blockSize = numWarps * WARPSIZE;
// Has to be 4 because of shared memory swizzling...could make it 2 with different swizzling
constexpr int kSlices = 4;
// To to global memory copies asynchronously or synchronously
constexpr bool sync = false;
constexpr bool rasterized = true;

using SharedSize = WarpMma::SharedSize;
constexpr int pipelineDepth = 2;
constexpr int maxPipelineDepth = 4;

/** The parameters required to run a search on the host.
 */
struct FindPairsParamsHost {
    double epsilon;  // The maximum distance between points to be considered a pair.
    Mma::mmaShape paddedSearchShape;  // The dimensions of the search data. Number of query
                                      // points, candidate points, and dimensions of each point.
    Mma::mmaShape inputSearchShape;   // The actual dimensions of the search data. Not including
                                      // padded values.
    Points::PointList<half_float::half> pointList;  // Host memory containing all points.
    bool skipPairs;                                 // Skip writing out pairs, useful for testing
    bool skipPointsDownload;  // Skip allocating and copying input points to the device. Useful
                              // to speed up testing if you are only changing epsilon.
    std::string outputPath;   // Filepath to write data to.
};

/**The parameters required to run a search on the device.
 */
struct FindPairsParamsDevice {
    float epsilonSquared;  // The maximum distance between points to be considered a pair.
    Mma::mmaShape paddedSearchShape;  // The dimensions of the search data. Number of query
                                      // points, candidate points, and dimensions of each point.
    Mma::mmaShape inputSearchShape;   // The actual dimensions of the search data. Not including
                                      // padded values.
    half2* queryPoints;               // Global memory array containing all query points.
    half2* candidatePoints;           // Global memory array containing all candidate points.
    float* sumSqQueries;              // Summed up squared dimensions of query points.
    float* sumSqCandidates;           // Summed up squared dimensions of Candidates points.
};

__host__ __device__ constexpr Mma::mmaShape GetBlockTileDims() {
    WarpMma::WarpTileDims warpDims = WarpMma::GetWarpTileDims();
    int m = numWarpRows * warpDims.m;
    int n = numWarpCols * warpDims.n;
    // Can buffer multiple k slices into shared memory at a time
    int k = kSlices * warpDims.k;
    return Mma::mmaShape{m, n, k};
}

constexpr Mma::mmaShape blockTileDims = GetBlockTileDims();

// Compute how much shared memory to allocate when we launch the kernel
constexpr int AElemsPerStage = blockTileDims.m * blockTileDims.k / WarpMma::dimPerInt4;
constexpr int BElemsPerStage = blockTileDims.n * blockTileDims.k / WarpMma::dimPerInt4;
constexpr int ElemsPerStage = AElemsPerStage + BElemsPerStage;

/** Get global coordinates of upper left element this block is responsible for. Converts a 1D
 * block index to a 2D coordinate that this block is responsible for computing.
 *
 * \param blockCoords 1D Array of coordinates each block in the grid is responsible for.
 *
 * \return Global coordinates of upper left element
 */
__device__ Coordinate GetBaseBlockCoordinate(Coordinate* blockCoords) {
    // Extract this block's 2D index from the 1D launch index.
    Coordinate coords = blockCoords[blockIdx.x];

    int baseRow = coords.row * GetBlockTileDims().m;
    int baseCol = coords.col * GetBlockTileDims().n;

    return {baseRow, baseCol};
}

/** Local coordinates of warp. Get local (relative to this block) coordinates of upper left
 * element that a warp in a block is responsible for
 *
 * \param warpId Index of warp to return coordinates for
 *
 * \return The local coordinates (relative to this block) of upper left
 * element
 */
__device__ Coordinate GetBaseLocalWarpCoordinate(int warpId) {
    int warpRow = warpId / numWarpCols;
    int warpCol = warpId % numWarpCols;
    int baseRow = warpRow * WarpMma::GetWarpTileDims().m;
    int baseCol = warpCol * WarpMma::GetWarpTileDims().n;
    return {baseRow, baseCol};
}

/** Global coordinates a warp is responsible for. Get global coordinates of upper left element
 * that a given warp in a block is responsible for
 *
 * \param baseBlockCoord Global coordinates of upper left element this block is
 * responsible for
 * \param warpId Index of warp to return coordinates for
 *
 * \return The global coordinates of upper left element
 */
__device__ Coordinate GetBaseWarpCoordinate(const Coordinate& baseBlockCoord, int warpId) {
    Coordinate baseLocal = GetBaseLocalWarpCoordinate(warpId);
    int baseRow = baseBlockCoord.row + baseLocal.row;
    int baseCol = baseBlockCoord.col + baseLocal.col;
    return {baseRow, baseCol};
}

/** Given an address, swizzle it to avoid bank conflicts. Takes in either an expected address in
 * row/column format, and swizzles it so that each individual copy has no conflicts with others.
 * This is highly specialized for an MMA operation.
 *
 *
 * \param sharedMemRow The row of shared memory to store into.
 * \param sharedMemColumn The column of shared memory to store into.
 * \param columnsPerRow How many columns shared memory has.
 *
 * \return
 */
__device__ int SwizzleAddress(int sharedMemRow, int sharedMemColumn, int columnsPerRow) {
    // Column ^ Row in shared memory
    int swizzledLane = sharedMemColumn ^ (sharedMemRow % columnsPerRow);
    int swizzledAddress = sharedMemRow * columnsPerRow + swizzledLane;
    return swizzledAddress;
}

/** Pages points from global to shared memory asynchronously. Pages in a certain number of
 points in global memory from a specified start point. Each thread does an int4 copy from global
 to shared for every point that the BlockTile needs. This method swizzles the addresses as it
 stores into shared memory to avoid shared memory conflicts.
 *
 * \param globalArray Base address of global memory array to page from
 * \param sharedArray Base address of shared memory array to page into
 * \param firstGlobalPoint Index of the first point in global memory to page in
 * \param numPoints How many points to page in
 * \param globalKStart The first k dimension to page in from global memory
 * \param globalKStride The number of dimensions per point in global memory
 *
 * \return
 */
__device__ void LoadGlobalToSharedAsync(cuda::pipeline<cuda::thread_scope_thread>& pipe,
                                        half2* globalArray, SharedSize* sharedArray,
                                        int firstGlobalPoint, int numPoints, int globalKStart,
                                        int globalKStride) {
    SharedSize* reinterpretedGlobalArray = reinterpret_cast<SharedSize*>(globalArray);
    // TODO make this use globalKStart
    static_assert(
        IsDivisionExact(GetBlockTileDims().k, WarpMma::dimPerInt4),
        "Block tile k dimension is not cleanly divisible by shared memory kGroup (int4) size");
    // A kGroup is a group of InPrec k values organized together for efficiency
    // How many int4 values must be copied to copy an entire point over
    constexpr int int4PerPoint = GetBlockTileDims().k / WarpMma::dimPerInt4;  // 8

    int firstPoint = threadIdx.x / int4PerPoint;
    int firstDim = threadIdx.x % int4PerPoint;
    int globalLeadingDim = globalKStride / WarpMma::dimPerInt4;
    int pointStride = blockSize / int4PerPoint;  // Block copies 16 points/iteration

    // First 8 threads copy over point 1, next 8 point 2, so on until all points are paged over
    for (int point = firstPoint; point < numPoints; point += pointStride) {
        // Extact int4 from global
        int globalPoint = point + firstGlobalPoint;
        int globalDim = firstDim + (globalKStart / WarpMma::dimPerInt4);
        int globalPointIndex = (globalPoint * globalLeadingDim) + globalDim;

        // Store int4 to shared
        int swizzledAddress = SwizzleAddress(point, firstDim, int4PerPoint);

        if (Debug && blockIdx.x == 0) {
            // To check for coalesced accesses
            printf("globalPointIndex T%d Point %d: %d\n", threadIdx.x, point, globalPointIndex);
            printf("swizzledAddress T%d Point %d: %d\n", threadIdx.x, point, swizzledAddress);
        }
        cuda::memcpy_async(sharedArray + swizzledAddress,
                           reinterpretedGlobalArray + globalPointIndex, sizeof(SharedSize), pipe);
    }
}

/** Pages points from global to shared memory. Pages in a certain number of points in global
 memory from a specified start point. Each thread does an int4 copy from global to
 shared for every point that the BlockTile needs. This method swizzles the addresses as it
 stores into shared memory to avoid shared memory conflicts.
 *
 * \param globalArray Base address of global memory array to page from
 * \param sharedArray Base address of shared memory array to page into
 * \param firstGlobalPoint Index of the first point in global memory to page in
 * \param numPoints How many points to page in
 * \param globalKStart The first k dimension to page in from global memory
 * \param globalKStride The number of dimensions per point in global memory
 *
 * \return
 */
__device__ void LoadGlobalToShared(half2* globalArray, SharedSize* sharedArray,
                                   int firstGlobalPoint, int numPoints, int globalKStart,
                                   int globalKStride) {
    // TODO make this use globalKStart
    SharedSize* reinterpretedGlobalArray = reinterpret_cast<SharedSize*>(globalArray);
    static_assert(
        IsDivisionExact(GetBlockTileDims().k, WarpMma::dimPerInt4),
        "Block tile k dimension is not cleanly divisible by shared memory kGroup (int4) size");
    // A kGroup is a group of InPrec k values organized together for efficiency
    constexpr int kGroupsPerPoint = GetBlockTileDims().k / WarpMma::dimPerInt4;  // 8

    int firstPoint = threadIdx.x / kGroupsPerPoint;
    int firstDim = threadIdx.x % kGroupsPerPoint;
    int globalLeadingDim = globalKStride / WarpMma::dimPerInt4;
    int pointStride = blockDim.x / kGroupsPerPoint;

    // First 8 threads copy over point 1, next 8 point 2, so on until all points are paged over
    for (int point = firstPoint; point < numPoints; point += pointStride) {
        // Extact int4 from global
        int globalPoint = point + firstGlobalPoint;
        int globalDim = firstDim + (globalKStart / WarpMma::dimPerInt4);
        int globalPointIndex = (globalPoint * globalLeadingDim) + globalDim;
        SharedSize values = reinterpretedGlobalArray[globalPointIndex];

        // Store int4 to shared
        int swizzledAddress = SwizzleAddress(point, firstDim, kGroupsPerPoint);
        sharedArray[swizzledAddress] = values;
    }
}

/** Accumulates a single k slice into a warp tile from shared memory. Data needs to be done
 * being paged in from global memory before calling this method.
 *
 *
 * \param warpTile The warp tile to accumulate into
 * \param ATile Pointer to start of shared memory array containing A
 * \param BTile Pointer to start of shared memory array containing B
 * \param baseLocalWarpCoord The upper left coordinate this warp is responsible for. Scoped to
 * the block (not global matrix).
 *
 * \return
 */
__device__ void AccumulateKSliceWarpTile(WarpMma::WarpTile& warpTile, SharedSize* ATile,
                                         SharedSize* BTile, const Coordinate& baseLocalWarpCoord) {
    // Accumulate into D as many times as we need to
    for (int kslice = 0; kslice < kSlices; kslice++) {
        warpTile.warpTileLoadA(ATile, kslice, blockTileDims.k, baseLocalWarpCoord.row);
        warpTile.warpTileLoadB(BTile, kslice, blockTileDims.k, baseLocalWarpCoord.col);
        warpTile.warpTileMma();
    }
}

/** Checks if an array of a specified type is an exact multiple of the desired alignment in
 * bytes.
 *
 * \param numElements The number of elements in the array
 * \param multiple The number of bytes to be a multiple of
 * \param arrayType The type of the array elements. Used to determine bytes.
 *
 * \return If the array length is an exact multiple.
 */
template <int multiple, typename arrayType>
__device__ constexpr bool IsMultiple(int numElements) {
    return (numElements * sizeof(arrayType)) % multiple == 0;
}

/** Return the pointer to the next tile based on the pipeline index. This is an optimization
 * because the compiler is putting the arrays in local memory becaus it can't determine the
 * constant indices.
 *
 * \param pipelineIndex The next index to load.
 * \param ATile The base address for the A Tile in shared memory.
 * \param BTile The base address fo the B Tile in shared memory.
 * \param nextATile Returns the next address in shared memory to read.
 * \param nextBTile Returns the next address in shared memory to read.
 *
 * \return void. Returns through pointers passed in
 */
__device__ void GetNextSharedTile(int pipelineIndex, SharedSize** ATile, SharedSize** BTile,
                                  SharedSize** nextATile, SharedSize** nextBTile) {
    switch (pipelineIndex) {
        case 0:
            *nextATile = ATile[0];
            *nextBTile = BTile[0];
            break;
        case 1:
            *nextATile = ATile[1];
            *nextBTile = BTile[1];
            break;
        case 2:
            *nextATile = ATile[2];
            *nextBTile = BTile[2];
            break;
        case 3:
            *nextATile = ATile[3];
            *nextBTile = BTile[3];
            break;
    }
}

/** Asynchronously loads all the necessary sums of squared points that the entire block will
 * need from global memory to shared memory. You must wait on the group before attempting to
 * read values from shared memory.
 *
 * \param pipe The cuda pipeline to commit transactions to.
 * \param globalSums The sums of the squared dimensions of the points in global memory.
 * \param sharedSums The sums of the squared dimensions of the points in global memory.
 * \param firstSum The index of the first sum this block needs to page in.
 * \param numSums How many sums the entire block needs to page in.
 *
 */
__device__ void LoadSumSquaredAsync(cooperative_groups::thread_block& group, float* globalSums,
                                    float* sharedSums, int firstSum, int numSums) {
    cooperative_groups::memcpy_async(group, sharedSums, &globalSums[firstSum],
                                     sizeof(float) * numSums);
}

/** Finds all pairs of query and candidate points within epsilon. Computes the distance using
 * the summed squared dimensions of each point in combination with a matrix multiplication of
 * the points.
 *
 * \param params See struct documentation.
 * \param blockCoords Array of coordinates specifying which chunk of the output matrix each
 * block is responsible for.
 *
 */
__global__ void FindPairsKernel(FindPairsParamsDevice params, Coordinate* blockCoords,
                                Pairs::Pairs pairs) {
    int warpId = threadIdx.x / 32;

    __align__(128) __shared__ float squaredQueries[blockTileDims.m];
    __align__(128) __shared__ float squaredCandidates[blockTileDims.n];

    // Pipeline init
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    __align__(128) extern __shared__ SharedSize sharedMem[];

    SharedSize* ATile[maxPipelineDepth];
    SharedSize* BTile[maxPipelineDepth];

    // We need 128 byte alignment here so each point ends up in it's own row of shared memory
    static_assert(IsMultiple<128, SharedSize>(AElemsPerStage),
                  "A single stage of A must be a multiple of 128 bytes");
    static_assert(IsMultiple<128, SharedSize>(BElemsPerStage),
                  "A single stage of B must be a multiple of 128 bytes");
    for (int i = 0; i < pipelineDepth; ++i) {
        int stageOffset = (i * ElemsPerStage);
        ATile[i] = sharedMem + stageOffset;
        BTile[i] = ATile[i] + AElemsPerStage;
        if (threadIdx.x == 0 && Debug) {
            printf("Shared Memory addresses in stage %d are:\nATile: %p\nBTile: %p\n", i, ATile[i],
                   BTile[i]);
        }
    }

    // Global MMA Scoped
    // Compute Upper left coordinate that this block is responsible for
    Coordinate baseBlockCoord = GetBaseBlockCoordinate(blockCoords);
    if (Debug && threadIdx.x == 0) {
        printf("baseBlockCoord block %d is %d, %d\n", blockIdx.x, baseBlockCoord.row,
               baseBlockCoord.col);
    }
    // Block MMA Scoped
    // Compute local warpBase Coordinates relative to this block. Useful for extracting the
    // appropriate values from shared memory
    Coordinate baseLocalWarpCoord = GetBaseLocalWarpCoordinate(warpId);
    // Global MMA Scoped
    // Compute the Upper left coordinate that each warp is responsible for
    // MMA Useful for performing final inspection of results
    Coordinate baseWarpCoord = GetBaseWarpCoordinate(baseBlockCoord, warpId);

    // Start transferring sums of squared points over
    auto group = cooperative_groups::this_thread_block();
    LoadSumSquaredAsync(group, params.sumSqQueries, squaredQueries, baseBlockCoord.row,
                        GetBlockTileDims().m);
    LoadSumSquaredAsync(group, params.sumSqCandidates, squaredCandidates, baseBlockCoord.col,
                        GetBlockTileDims().n);

    // Create numWarps warpTiles to process what this block is responsible for
    WarpMma::WarpTile warpTile;
    warpTile.clearD();

    // All threads colloborate to move Global-->Shared as data in shared is shared between all
    // warps in the block

    // Normal copy global->register, register->shared
    // Single buffered
    if (sync) {
        for (int kStart = 0; kStart < params.paddedSearchShape.k; kStart += GetBlockTileDims().k) {
            __syncthreads();  // Wait for all warps to complete using data
            // Page A in
            LoadGlobalToShared(params.queryPoints, ATile[0], baseBlockCoord.row,
                               GetBlockTileDims().m, kStart, params.paddedSearchShape.k);

            // Page B in
            LoadGlobalToShared(params.candidatePoints, BTile[0], baseBlockCoord.col,
                               GetBlockTileDims().n, kStart, params.paddedSearchShape.k);
            __syncthreads();  // Wait for all data to be paged before computing

            // Debug print statements
            if (Debug) {
                if (threadIdx.x == 0) {
                    PrintMatrix<half>("Shared A", reinterpret_cast<half*>(ATile),
                                      GetBlockTileDims().m, GetBlockTileDims().k);
                    PrintMatrix<half>("Shared B", reinterpret_cast<half*>(BTile),
                                      GetBlockTileDims().n, GetBlockTileDims().k);
                }
                __syncthreads();
            }

            AccumulateKSliceWarpTile(warpTile, ATile[0], BTile[0], baseLocalWarpCoord);
        }
    }
    // Async pipeline direct global->shared
    else {
        // Fill pipeline
        // Keep track of which k index to load next since it's different than the next k we need
        // to compute with
        int nextKToLoad = 0;
        for (int i = 0; i < pipelineDepth; ++i) {
            // Always acquire and commit since wait_prior< > takes in a constant.
            pipeline.producer_acquire();

            if (nextKToLoad < params.paddedSearchShape.k) {
                // Page A in
                LoadGlobalToSharedAsync(pipeline, params.queryPoints, ATile[i], baseBlockCoord.row,
                                        blockTileDims.m, nextKToLoad, params.paddedSearchShape.k);

                // Page B in
                LoadGlobalToSharedAsync(pipeline, params.candidatePoints, BTile[i],
                                        baseBlockCoord.col, blockTileDims.n, nextKToLoad,
                                        params.paddedSearchShape.k);
                nextKToLoad += blockTileDims.k;
            }
            pipeline.producer_commit();
        }
        // Pipeline stage to consume next
        int pipelineIndex = 0;
        for (int kStart = 0; kStart < params.paddedSearchShape.k; kStart += blockTileDims.k) {
            SharedSize *nextATile, *nextBTile;
            GetNextSharedTile(pipelineIndex, ATile, BTile, &nextATile, &nextBTile);

            cuda::pipeline_consumer_wait_prior<pipelineDepth - 1>(pipeline);

            // Thread scoped pipeline so must sync so we can read other thread's data
            __syncthreads();

            AccumulateKSliceWarpTile(warpTile, nextATile, nextBTile, baseLocalWarpCoord);

            __syncthreads();

            pipeline.consumer_release();

            // Queue up the next stage of the pipeline
            pipeline.producer_acquire();
            // Still queue up empty stage if all the data we need is in the pipeline so wait
            // doesn't block indefinitely at the end of the computation
            if (nextKToLoad < params.paddedSearchShape.k) {
                // Page A in
                LoadGlobalToSharedAsync(pipeline, params.queryPoints, nextATile, baseBlockCoord.row,
                                        blockTileDims.m, nextKToLoad, params.paddedSearchShape.k);

                // Page B in
                LoadGlobalToSharedAsync(pipeline, params.candidatePoints, nextBTile,
                                        baseBlockCoord.col, blockTileDims.n, nextKToLoad,
                                        params.paddedSearchShape.k);

                nextKToLoad += blockTileDims.k;
            }
            pipeline.producer_commit();

            // Set up next iteration to load from next stage in shared mem.
            pipelineIndex = (pipelineIndex + 1) % pipelineDepth;
        }
    }
    // Wait for sums of squared terms to complete transferring
    cooperative_groups::wait(group);
    if (threadIdx.x == 0 && Debug == true) {
        for (int i = 0; i < blockTileDims.n; ++i) {
            printf("Shared Query Sums %d: %f\n", i, squaredQueries[i]);
            printf("Shared Candidate Sums %d: %f\n", i, squaredCandidates[i]);
        }
    }

    // Inspect the final results to see if we are within epsilon
    warpTile.inspectResults(baseBlockCoord, baseWarpCoord, squaredQueries, squaredCandidates,
                            params.epsilonSquared, params.inputSearchShape, pairs);
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

    // Release global memory before allocating over the top of it, if it has already been allocated.
    releaseGlobalMemory();

    gpuErrchk(cudaMalloc(d_Query, querySize));
    gpuErrchk(cudaMalloc(d_Candidate, candidateSize));

    gpuErrchk(cudaMemcpy(*d_Query, h_Query.data(), querySize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(*d_Candidate, h_Candidate.data(), candidateSize, cudaMemcpyHostToDevice));
}

/** Transfer a set of input points to global memory on the device for processing. The memory
 * allocated here must be released later with the "releaseGlobalMemory()" call.
 *
 */
template <typename Precision>
__host__ void allocateTransferPoints(const std::vector<Precision>& h_Query,
                                     const std::vector<Precision>& h_Candidate) {
    TransferPointsToGMem(h_Query, h_Candidate, &d_AValues, &d_BValues);
}

/** Given a set of input points, finds all pairs of points in the query and candidate points.
 * Launches the required kernels to do so.
 *
 * \param hostParams See struct documentation.
 *
 * \return The results of the search.
 */
__host__ Results FindPairs(const FindPairsParamsHost& hostParams) {
    // Push points from host to the GPU.
    if (!hostParams.skipPointsDownload) {
        allocateTransferPoints(hostParams.pointList.values, hostParams.pointList.values);
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaEvent_t squaredSumsStart, squaredSumsStop, findPairsStop;
    cudaEventCreate(&squaredSumsStart);
    cudaEventCreate(&squaredSumsStop);
    cudaEventCreate(&findPairsStop);

    cudaEventRecord(squaredSumsStart, 0);

    // Compute sums of squared dimensions
    using sumSize = float;
    sumSize *d_ASqSums, *d_BSqSums;
    d_ASqSums = SumSqd::ComputeSquaredSums<sumSize>(d_AValues, hostParams.paddedSearchShape.m,
                                                    hostParams.paddedSearchShape.k);
    d_BSqSums = SumSqd::ComputeSquaredSums<sumSize>(d_BValues, hostParams.paddedSearchShape.n,
                                                    hostParams.paddedSearchShape.k);

    cudaEventRecord(squaredSumsStop, 0);

    // Allocate place to store pairs to.
    // Assume a certain amount of pairs will be found, 1000 is a fair guess without overflowing
    // memory.
    size_t expectedPairs = hostParams.inputSearchShape.m * 1000;
    Pairs::Pairs pairs(expectedPairs);
    pairs.init();

    // Determine thread block launch parameters
    size_t numBlocksRow = ceil(1.0 * hostParams.paddedSearchShape.m / GetBlockTileDims().m);
    size_t numBlocksCol = ceil(1.0 * hostParams.paddedSearchShape.n / GetBlockTileDims().n);
    dim3 gridDim(numBlocksRow * numBlocksCol, 1, 1);
    dim3 blockDim(blockSize, 1, 1);
    size_t sharedMemBytes = pipelineDepth * ElemsPerStage * sizeof(SharedSize);

    // Generate rasterized thread block layout for better cache locality
    std::vector<Coordinate> h_blockCoords =
        rasterized ? Raster::RasterizeLayout(10, 10, numBlocksRow, numBlocksCol)
                   : Raster::ConventionalLayout(numBlocksRow, numBlocksCol);
    Coordinate* d_blockCoords;
    size_t size = sizeof(Coordinate) * h_blockCoords.size();
    gpuErrchk(cudaMalloc(&d_blockCoords, size));
    gpuErrchk(cudaMemcpy(d_blockCoords, h_blockCoords.data(), size, cudaMemcpyHostToDevice));

    if (Debug) {
        printf("Requesting %lu bytes of shared memory\n", sharedMemBytes);
        printf("Block Size is %d x %d x %d\n", blockTileDims.m, blockTileDims.n, blockTileDims.k);
    }

    gpuErrchk(cudaFuncSetAttribute(FindPairsKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   sharedMemBytes));

    // Build up parameters to run on the device.
    float epsilonSquared = hostParams.epsilon * hostParams.epsilon;
    auto deviceParams = SimSearch::FindPairsParamsDevice{epsilonSquared,
                                                         hostParams.paddedSearchShape,
                                                         hostParams.inputSearchShape,
                                                         d_AValues,
                                                         d_BValues,
                                                         d_ASqSums,
                                                         d_BSqSums};
    // Run the actual search kernel
    FindPairsKernel<<<gridDim, blockDim, sharedMemBytes>>>(deviceParams, d_blockCoords, pairs);

    gpuErrchk(cudaEventRecord(findPairsStop, 0));
    // Synchronize then sort pairs and save them of
    gpuErrchk(cudaDeviceSynchronize());
    double sortStartTime = omp_get_wtime();
    if (!hostParams.skipPairs) {
        pairs.sort();
        // Open file stream and write data to it
        std::string pairsPath = hostParams.outputPath + ".pairs";
        std::cout << "Output file is: " << pairsPath << std::endl;
        std::ofstream outFile(pairsPath);
        outFile << pairs;
    }
    unsigned long long pairsFound = pairs.getPairsFound();
    unsigned long long pairsStored = pairs.getPairsStored();
    pairs.release();
    double sortEndTime = omp_get_wtime();

    gpuErrchk(cudaEventSynchronize(findPairsStop));

    // Compute result statistics
    float elapsedTime, sumSquaredTime, findPairsTime, sortTime;
    cudaEventElapsedTime(&elapsedTime, squaredSumsStart, findPairsStop);
    cudaEventElapsedTime(&sumSquaredTime, squaredSumsStart, squaredSumsStop);
    cudaEventElapsedTime(&findPairsTime, squaredSumsStop, findPairsStop);
    sortTime = sortEndTime - sortStartTime;
    elapsedTime /= 1000;
    sumSquaredTime /= 1000;
    findPairsTime /= 1000;

    printf("Total Kernel Elapsed time: %f seconds\n", elapsedTime);
    printf("SumSquard Kernel Elapsed time: %f seconds\n", sumSquaredTime);
    printf("FindPairs Kernel Elapsed time: %f seconds\n", findPairsTime);
    printf("Pairs sort Elapsed time: %f seconds\n", sortTime);
    std::cout << "Pairs found: " << pairsFound << std::endl;
    // Estimated TFLOPS that we computed. Don't count padded 0's as useful computation.
    const float tflops = static_cast<float>(hostParams.inputSearchShape.m) *
                         hostParams.inputSearchShape.n * hostParams.inputSearchShape.k * 2 /
                         elapsedTime / 1e12;
    printf("Estimated TFLOPS %.3f\n", tflops);

    // Release device memory resources
    cudaEventDestroy(squaredSumsStart);
    cudaEventDestroy(squaredSumsStop);
    cudaEventDestroy(findPairsStop);
    cudaFree(d_ASqSums);
    cudaFree(d_BSqSums);
    cudaFree(d_blockCoords);

    return Results{tflops, pairsFound, pairsStored, hostParams.inputSearchShape,
                   hostParams.paddedSearchShape};
}

__host__ void releaseGlobalMemory() {
    if (d_AValues) {
        cudaFree(d_AValues);
        d_AValues = nullptr;
    }

    if (d_BValues) {
        cudaFree(d_BValues);
        d_BValues = nullptr;
    }
}

};  // namespace SimSearch

#endif /* end of include guard: FINDPAIRS_CUH_KP5RAZNA */
