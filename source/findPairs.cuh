/******************************************************************************
1* File:             findPairs.cuh
 * * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      All the functionality to compute a single blockTile
 *****************************************************************************/

#ifndef BLOCKMMA_CUH_KP5RAZNA
#define BLOCKMMA_CUH_KP5RAZNA

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda/pipeline>

#include "ptxMma.cuh"
#include "utils.cuh"
#include "warpMma.cuh"

namespace BlockTile {

// Block Parameters
constexpr int numWarpCols = 2;
constexpr int numWarpRows = 2;
constexpr int numWarps = numWarpCols * numWarpRows;
constexpr int kSlices = 4;
constexpr int coarseFactor = 1;
// To to global memory copies asynchronously or synchronously
constexpr bool sync = false;

using SharedSize = WarpMma::SharedSize;
constexpr int pipelineDepth = 2;
constexpr int maxPipelineDepth = 4;

struct FindPairsParams {
    float epsilonSquared;       // The maximum distance between points to be considered a pair.
    Mma::mmaShape searchShape;  // The dimensions of the search data. Number of query points,
                                // candidate points, and dimensions of each point.
    unsigned long long* iterationCount;  // TODO Delete this
    half2* queryPoints;                  // Global memory array containing all query points.
    half2* candidatePoints;              // Global memory array containing all candidate points.
    float* sumSqQueries;                 // Summed up squared dimensions of query points.
    float* sumSqCandidates;              // Summed up squared dimensions of Candidates points.
};

__host__ __device__ constexpr Mma::mmaShape GetBlockTileDims() {
    WarpMma::WarpTileDims warpDims = WarpMma::GetWarpTileDims();
    int m = numWarpRows * warpDims.m;
    int n = numWarpCols * warpDims.n;
    // Can buffer multiple k slices into shared memory at a time
    int k = kSlices * warpDims.k;
    return Mma::mmaShape{m, n, k};
}

constexpr BlockTileDims blockTileDims = GetBlockTileDims();

// Compute how much shared memory to allocate when we launch the kernel
constexpr int AElemsPerStage = blockTileDims.m * blockTileDims.k / WarpMma::dimPerInt4;
constexpr int BElemsPerStage = blockTileDims.n * blockTileDims.k / WarpMma::dimPerInt4;
constexpr int ElemsPerStage = AElemsPerStage + BElemsPerStage;

/** Get global coordinates of upper left element this block is responsible for.
 *
 * \return Global coordinates of upper left element
 */
__device__ Mma::Coordinate GetBaseBlockCoordinate() {
    int baseRow = blockIdx.y * GetBlockTileDims().m;
    int baseCol = blockIdx.x * GetBlockTileDims().n;
    return {baseRow, baseCol};
}

/** Local coordinates of warp. Get local (relative to this block) coordinates of upper left element
 * that a warp in a block is responsible for
 *
 * \param warpId Index of warp to return coordinates for
 *
 * \return The local coordinates (relative to this block) of upper left
 * element
 */
__device__ Mma::Coordinate GetBaseLocalWarpCoordinate(int warpId) {
    int warpRow = warpId / numWarpCols;
    int warpCol = warpId % numWarpCols;
    int baseRow = warpRow * WarpMma::GetWarpTileDims().m;
    int baseCol = warpCol * WarpMma::GetWarpTileDims().n;
    return {baseRow, baseCol};
}

/** Global coordinates a warp is responsible for. Get global coordinates of upper left element that
 * a given warp in a block is responsible for
 *
 * \param baseBlockCoord Global coordinates of upper left element this block is
 * responsible for
 * \param warpId Index of warp to return coordinates for
 *
 * \return The global coordinates of upper left element
 */
__device__ Mma::Coordinate GetBaseWarpCoordinate(Mma::Coordinate baseBlockCoord, int warpId) {
    Mma::Coordinate baseLocal = GetBaseLocalWarpCoordinate(warpId);
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

/** Pages points from global to shared memory asynchronously. Pages in a certain number of points in
 global memory from a specified start point. Each thread does an int4 copy from global to shared for
 every point that the BlockTile needs. This method swizzles the addresses as it stores into shared
 memory to avoid shared memory conflicts.
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
    int pointStride = blockDim.x / int4PerPoint;  // Block copies 16 points/iteration

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
 shared for every point that the BlockTile needs. This method swizzles the addresses as it stores
 into shared memory to avoid shared memory conflicts.
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

/** Accumulates a single k slice into a warp tile from shared memory. Data needs to be done being
 * paged in from global memory before calling this method.
 *
 *
 * \param warpTile The warp tile to accumulate into
 * \param baseLocalWarpCoord The upper left coordinate this warp is responsible for. Scoped to the
 * block (not global matrix).
 * \param kBlockTile k dimension of the block tile.
 * \param ATile Pointer to start of shared memory array containing A
 * \param BTile Pointer to start of shared memory array containing B
 *
 * \return
 */
__device__ void AccumulateKSliceWarpTile(WarpMma::WarpTile& warpTile, SharedSize* ATile,
                                         SharedSize* BTile, Mma::Coordinate& baseLocalWarpCoord,
                                         int kBlockTile) {
    // Accumulate into D as many times as we need to
    for (int kslice = 0; kslice < kSlices; kslice++) {
        warpTile.warpTileLoadA(ATile, kslice, blockTileDims.k, baseLocalWarpCoord.row);
        warpTile.warpTileLoadB(BTile, kslice, blockTileDims.k, baseLocalWarpCoord.col);
        warpTile.warpTileMma();
    }
}

/** Checks if an array of a specified type is an exact multiple of the desired alignment in bytes.
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

/** Return the pointer to the next tile based on the pipeline index. This is an optimization because
 * the compiler is putting the arrays in local memory becaus it can't determine the constant
 * indices.
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
 * need from global memory to shared memory. You must wait on the group before attempting to read
 * values from shared memory.
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
 *
 */
__global__ void FindPairsKernel(FindPairsParams params) {
    int warpId = threadIdx.x / 32;
    // Kind of a useless count to get compiler to not optimize away my code
    unsigned int count = 0;

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

    __shared__ unsigned long long blockCount;
    if (threadIdx.x == 0) {
        blockCount = 0;
    }
    __syncthreads();

    // Global MMA Scoped
    // Compute Upper left coordinate that this block is responsible for
    Mma::Coordinate baseBlockCoord = GetBaseBlockCoordinate();
    if (Debug && threadIdx.x == 0) {
        printf("baseBlockCoord.row T%d is %d\n", threadIdx.x, baseBlockCoord.row);
        printf("baseBlockCoord.col T%d is %d\n", threadIdx.x, baseBlockCoord.col);
    }
    // Block MMA Scoped
    // Compute local warpBase Coordinates relative to this block. Useful for extracting the
    // appropriate values from shared memory
    Mma::Coordinate baseLocalWarpCoord = GetBaseLocalWarpCoordinate(warpId);
    // Global MMA Scoped
    // Compute the Upper left coordinate that each warp is responsible for
    // MMA Useful for performing final inspection of results
    Mma::Coordinate baseWarpCoord = GetBaseWarpCoordinate(baseBlockCoord, warpId);

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
        for (int kStart = 0; kStart < params.searchShape.k; kStart += GetBlockTileDims().k) {
            __syncthreads();  // Wait for all warps to complete using data
            // Page A in
            LoadGlobalToShared(params.queryPoints, ATile[0], baseBlockCoord.row,
                               GetBlockTileDims().m, kStart, params.searchShape.k);

            // Page B in
            LoadGlobalToShared(params.candidatePoints, BTile[0], baseBlockCoord.col,
                               GetBlockTileDims().n, kStart, params.searchShape.k);
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

            AccumulateKSliceWarpTile(warpTile, ATile[0], BTile[0], baseLocalWarpCoord,
                                     blockTileDims.k);
        }
    }
    // Async pipeline direct global->shared
    else {
        // Fill pipeline
        // Keep track of which k index to load next since it's different than the next k we need
        // to compute with
        int nextKToLoad = 0;
        // Handles situation where pipeline is deeper than we can fill with input data
        int numStagesToBuffer = min(pipelineDepth, params.searchShape.k / blockTileDims.k);
        for (int i = 0; i < numStagesToBuffer; ++i) {
            pipeline.producer_acquire();
            if (blockIdx.x == 0 && threadIdx.x == 0 && Debug) {
                printf("Next k to load %d\n", nextKToLoad);
            }
            // Page A in
            LoadGlobalToSharedAsync(pipeline, params.queryPoints, ATile[i], baseBlockCoord.row,
                                    blockTileDims.m, nextKToLoad, params.searchShape.k);

            // Page B in
            LoadGlobalToSharedAsync(pipeline, params.candidatePoints, BTile[i], baseBlockCoord.col,
                                    blockTileDims.n, nextKToLoad, params.searchShape.k);
            nextKToLoad += blockTileDims.k;
            pipeline.producer_commit();
        }
        // Pipeline stage to consume next
        int pipelineIndex = 0;
        for (int kStart = 0; kStart < params.searchShape.k; kStart += blockTileDims.k) {
            if (blockIdx.x == 0 && threadIdx.x == 0 && Debug) {
                printf("Waiting for pipeline to compute k chunk %d\n", kStart);
                printf("Pipeline Index %d\n", pipelineIndex);
                printf("Next k to load %d\n", nextKToLoad);
            }
            SharedSize *nextATile, *nextBTile;
            GetNextSharedTile(pipelineIndex, ATile, BTile, &nextATile, &nextBTile);

            cuda::pipeline_consumer_wait_prior<pipelineDepth - 1>(pipeline);

            // Thread scoped pipeline so must sync so we can read other thread's data
            __syncthreads();

            AccumulateKSliceWarpTile(warpTile, nextATile, nextBTile, baseLocalWarpCoord,
                                     blockTileDims.k);

            __syncthreads();

            pipeline.consumer_release();

            // Queue up the next stage of the pipeline
            pipeline.producer_acquire();
            // Still queue up empty stage if all the data we need is in the pipeline so wait
            // doesn't block indefinitely at the end of the computation
            if (nextKToLoad < params.searchShape.k) {
                // Page A in
                LoadGlobalToSharedAsync(pipeline, params.queryPoints, nextATile, baseBlockCoord.row,
                                        blockTileDims.m, nextKToLoad, params.searchShape.k);

                // Page B in
                LoadGlobalToSharedAsync(pipeline, params.candidatePoints, nextBTile,
                                        baseBlockCoord.col, blockTileDims.n, nextKToLoad,
                                        params.searchShape.k);

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

    // Number within epsilon in each thread
    count += warpTile.inspectResults(baseBlockCoord, baseWarpCoord, squaredQueries,
                                     squaredCandidates, params.epsilonSquared);

    // Simple reduction in shared memory
    atomicAdd(&blockCount, count);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(params.iterationCount, blockCount);
    }
}

/** Finds all pairs of points in the query and candidate points. Launches the required kernels
 * to do so.
 *
 * \param params See struct documentation.
 *
 * \return
 */
__host__ void FindPairs(FindPairsParams params) {
    dim3 gridDim(ceil(1.0 * params.searchShape.n / GetBlockTileDims().n),
                 ceil(1.0 * params.searchShape.m / GetBlockTileDims().m), 1);
    dim3 blockDim(numWarps * WARPSIZE, 1, 1);
    size_t sharedMemBytes = pipelineDepth * ElemsPerStage * sizeof(SharedSize);

    if (Debug) {
        printf("Requesting %lu bytes of shared memory\n", sharedMemBytes);
    }

    gpuErrchk(cudaFuncSetAttribute(FindPairsKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   sharedMemBytes));

    FindPairsKernel<<<gridDim, blockDim, sharedMemBytes>>>(params);
}
};  // namespace BlockTile

#endif /* end of include guard: BLOCKMMA_CUH_KP5RAZNA */
