/******************************************************************************
 * File:             blockMma.cuh
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      All the functionality to compute a single blockTile
 *****************************************************************************/

#ifndef BLOCKMMA_CUH_KP5RAZNA
#define BLOCKMMA_CUH_KP5RAZNA

#include "utils.cuh"
#include "warpMma.cuh"

namespace BlockMma {

// Block Parameters
constexpr int numWarpCols = 2;
constexpr int numWarpRows = 2;
constexpr int numWarps = numWarpCols * numWarpRows;
constexpr int blockSize = WARPSIZE * numWarps;
constexpr int kSlices = 4;
constexpr int coarseFactor = 1;

using SharedSize = WarpMma::SharedSize;

struct BlockTileDims {
    int m{};
    int n{};
    int k{};
};

__host__ __device__ constexpr BlockTileDims GetBlockTileDims() {
    WarpMma::WarpTileDims warpDims = WarpMma::GetWarpTileDims();
    int m = numWarpRows * warpDims.m;
    int n = numWarpCols * warpDims.n;
    // Can buffer multiple k slices into shared memory at a time
    int k = kSlices * warpDims.k;
    return BlockTileDims{m, n, k};
}

// Compute how much shared memory to allocate when we launch the kernel
constexpr BlockMma::BlockTileDims blockTileDims = GetBlockTileDims();

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
__device__ void LoadGlobalToShared(SharedSize* globalArray, SharedSize* sharedArray,
                                   int firstGlobalPoint, int numPoints, int globalKStart,
                                   int globalKStride) {
    // TODO make this use globalKStart
    static_assert(
        IsDivisionExact(GetBlockTileDims().k, WarpMma::dimPerInt4),
        "Block tile k dimension is not cleanly divisible by shared memory kGroup (int4) size");
    // A kGroup is a group of InPrec k values organized together for efficiency
    constexpr int kGroupsPerPoint = GetBlockTileDims().k / WarpMma::dimPerInt4;  // 8

    int firstPoint = threadIdx.x / kGroupsPerPoint;
    int firstDim = threadIdx.x % kGroupsPerPoint;
    int globalLeadingDim = globalKStride / WarpMma::dimPerInt4;
    constexpr int pointStride = blockSize / kGroupsPerPoint;

    // First 8 threads copy over point 1, next 8 point 2, so on until all points are paged over
    for (int point = firstPoint; point < numPoints; point += pointStride) {
        // Extact int4 from global
        int globalPoint = point + firstGlobalPoint;
        int globalDim = firstDim + (globalKStart / WarpMma::dimPerInt4);
        int globalPointIndex = (globalPoint * globalLeadingDim) + globalDim;
        SharedSize values = globalArray[globalPointIndex];

        // Store int4 to shared
        int swizzledAddress = SwizzleAddress(point, firstDim, kGroupsPerPoint);
        sharedArray[swizzledAddress] = values;
    }
}

/** Computes an mma operation at the block scope.
 *
 * \param iterationCount TODO Delete this
 * \param AValues Global memory array containing elements of A
 * \param BValues Global memory array containing elements of B
 * \param globalKStride Global k dimension of MMA
 *
 */
__device__ void BlockTileMma(unsigned long long* iterationCount, SharedSize* AValues,
                             SharedSize* BValues, int globalKStride) {
    int tidx = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    // Kind of a useless count to get compiler to not optimize away my code
    unsigned int count = 0;

    // We need 128 byte alignment here so each point ends up in it's own row of shared memory
    __shared__ __align__(128)
        SharedSize ATile[GetBlockTileDims().m * GetBlockTileDims().k / WarpMma::dimPerInt4];
    __shared__ __align__(128)
        SharedSize BTile[GetBlockTileDims().n * GetBlockTileDims().k / WarpMma::dimPerInt4];

    // Global MMA Scoped
    // Compute Upper left coordinate that this block is responsible for
    Mma::Coordinate baseBlockCoord = GetBaseBlockCoordinate();
    // Block MMA Scoped
    // Compute local warpBase Coordinates relative to this block. Useful for extracting the
    // appropriate values from shared memory
    Mma::Coordinate baseLocalWarpCoord = GetBaseLocalWarpCoordinate(warpId);
    // Global MMA Scoped
    // Compute the Upper left coordinate that each warp is responsible for
    // MMA Useful for performing final inspection of results
    Mma::Coordinate baseWarpCoord = GetBaseWarpCoordinate(baseBlockCoord, warpId);

    // Create numWarps warpTiles to process what this block is responsible for
    WarpMma::WarpTile warpTile;
    warpTile.clearD();

    // All threads colloborate to move Global-->Shared as data in shared is shared between all warps
    for (int kStart = 0; kStart < globalKStride; kStart += GetBlockTileDims().k) {
        // in the block
        // Page A in
        __syncthreads();  // Wait for all warps to complete using data
        LoadGlobalToShared(AValues, ATile, baseBlockCoord.row, GetBlockTileDims().m, kStart,
                           globalKStride);

        // Page B in
        LoadGlobalToShared(BValues, BTile, baseBlockCoord.col, GetBlockTileDims().n, kStart,
                           globalKStride);
        __syncthreads();  // Wait for all data to be paged before computing

        if (threadIdx.x == 0) {
            PrintMatrix("Shared A", reinterpret_cast<half*>(ATile), GetBlockTileDims().m,
                        GetBlockTileDims().k);
            PrintMatrix("Shared B", reinterpret_cast<half*>(BTile), GetBlockTileDims().n,
                        GetBlockTileDims().k);
        }
        __syncthreads();

        // Accumulate into D as many times as we need to
        for (int kslice = 0; kslice < BlockMma::kSlices; kslice++) {
            warpTile.warpTileLoadA(ATile, kslice, blockTileDims.k, baseLocalWarpCoord.row);
            warpTile.warpTileLoadB(BTile, kslice, blockTileDims.k, baseLocalWarpCoord.col);
            warpTile.warpTileMma();
        }
    }
    warpTile.inspectResults(baseWarpCoord);

    // TODO the warpTile should determime if values are in bounds
    // This is all here so everything isn't optimized away
    for (int i = 0; i < WarpMma::numDFragments; i++) {
        if (tidx == 0 && (warpTile.D[i].Registers[0] > 10.0f)) {
            count++;
        }
    }

    if (tidx == 0) {
        atomicAdd(iterationCount, count);
    }
}

};  // namespace BlockMma

#endif /* end of include guard: BLOCKMMA_CUH_KP5RAZNA */
