/******************************************************************************
 * File:             warpMma.cuh
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      All the functionality to compute a single warp tile.
 *****************************************************************************/

#ifndef WARPMMA_CUH_OL9KOX7Y
#define WARPMMA_CUH_OL9KOX7Y

#include "matrix.cuh"
// #include "pair.cuh"
#include "ptxMma.cuh"
#include "utils.cuh"

// A single warp operation made up of many fragments of A, B, and D
namespace WarpMma {
using Coordinate = matrix::Coordinate;
using InPrec = Mma::InPrec;

// Warp  Parameters
constexpr int numAFragments = 4;
constexpr int numBFragments = 8;
constexpr int numDFragments = numAFragments * numBFragments;
// Swizzle the 8 columns of shared memory
constexpr int swizzleFactor = 8;
using SharedSize = int4;

static_assert(IsDivisionExact(sizeof(SharedSize), sizeof(InPrec)),
              "Input data precision doesn't divide cleanly into shared memory array");
constexpr int dimPerInt4 = sizeof(SharedSize) / sizeof(InPrec);

struct WarpTileDims {
    int m{};
    int n{};
    int k{};
};

// TODO Rounding might not work out here, have to be careful
constexpr int numRegistersA =
    Mma::dims.m * Mma::dims.k * sizeof(InPrec) / sizeof(uint32_t) / WARPSIZE;
constexpr int numRegistersB =
    Mma::dims.k * Mma::dims.n * sizeof(InPrec) / sizeof(uint32_t) / WARPSIZE;
constexpr int numRegistersD =
    Mma::dims.m * Mma::dims.n * sizeof(float) / sizeof(uint32_t) / WARPSIZE;

__host__ __device__ constexpr WarpTileDims GetWarpTileDims() {
    int m = numAFragments * Mma::dims.m;
    int n = numBFragments * Mma::dims.n;
    // Warp handles a single k slice at a time in registers
    int k = Mma::dims.k;
    return WarpTileDims{m, n, k};
}

// Each WarpTile stores multiple operands A and B to compute a large output matrix D
struct WarpTile {
   public:
    Mma::Fragment<uint32_t, numRegistersA> A[numAFragments]{};
    Mma::Fragment<uint32_t, numRegistersB> B[numBFragments]{};
    Mma::Fragment<float, numRegistersD> D[numDFragments]{};

    // Given an WarpTile, clears the D registers to 0.0. Useful for when starting a computation
    __device__ void clearD() {
        for (int i = 0; i < numDFragments; i++) {
            D[i].clear();
        }
    }

    /** Determine which chunk of dimensions to load for A. First m threads all read first 8
     * dimensions. Rest of the threads read the last 8 dimensions.
     *
     * \param laneId The thread's lane id (0,31)
     *
     * \return Which chunk of 8 dimensions a given thread should load
     */
    __device__ static int computeADimChunk(const int laneId) {
        return (laneId < Mma::dims.m ? 0 : 1);
    }

    /** Determine which chunk of dimensions to load for B. First n threads all read first 8
     * dimensions. The next 8 threads read the last 8 dimensions. Threads 16-31 must read the same
     * addresses as the lower 16 threads for the underlying ldmatrix ptx instruction to work.
     *
     * \param laneId The thread's lane id (0,31)
     *
     * \return Which chunk of 8 dimensions a given thread should load
     */
    __device__ static int computeBDimChunk(const int laneId) {
        return (laneId % 16 < Mma::dims.n ? 0 : 1);
    }

    using ColOffset = int (*)(int);

    /** The swizzled address of an int4 chunk of shared memory that a thread needs to read and copy
     * into shared memory via ldmatrix ptx.
     *
     * \param kSlice k slice to accumulate ex. (0..3)
     * \param kStride Shared mem. k dimension
     // TODO Is there a better way to phrase this variable?
     * \param fragmentRow First row of data in shared memory corresponding to a fragment
     * \param outerDims The number of rows of A or columns of B that will be read.
     * \param colOffset Given a lane Id, computes which chunk of 8 dimensions to read
     *
     * \return The swizzled shared memory address in an int4 array
     */
    __device__ int computeSwizzledIndex(const int kSlice, const int kStride, const int fragmentRow,
                                        const int outerDims, const ColOffset colOffset) {
        int laneId = threadIdx.x % WARPSIZE;
        int kStrideInt4 = kStride / dimPerInt4;
        int threadRow = fragmentRow + (laneId % outerDims);

        // Determine which chunk of dimensions we will load
        // base slice offset plus an additional offset for last 8 dimensions
        int threadCol = kSlice * Mma::dims.k / dimPerInt4 + colOffset(laneId);
        int swizzleRow = laneId % swizzleFactor;
        int swizzledCol = threadCol ^ swizzleRow;

        return threadRow * kStrideInt4 + swizzledCol;
    }

    /** loads a warptile's a fragments from relevant parts of shared memory. A is laid out in
     * row-major format in shared memory.
     *
     * \param aSharedAddr Start of shared mem. array
     * \param kslice k slice to accumulate ex. (0..3)
     * \param kStride Shared mem. k dimension
     * \param warpRow First row of A to load
     *
     */
    __device__ void warpTileLoadA(SharedSize* aSharedAddr, const int kslice, const int kStride,
                                  const int warpRow) {
        // Page fragment into A.
        Coordinate warpBase{warpRow, 0};
        for (int i = 0; i < numAFragments; i++) {
            // Determine which row we will load
            int fragmentRow = GetBaseFragmentCoordinate(warpBase, i, 0).row;
            int linearizedIndex =
                computeSwizzledIndex(kslice, kStride, fragmentRow, Mma::dims.m, computeADimChunk);
            Mma::loadAMatrix_16_16(&aSharedAddr[linearizedIndex], A[i]);
        }
    }

    /** loads a warptile's b fragments from relevant parts of shared memory. B is laid out in
     * row-major format in shared memory and is automatically transposed by the mma operation.
     *
     * \param bSharedAddr Start of shared mem. array
     * \param kslice k slice to accumulate ex. (0..3)
     * \param kStride Shared mem. k dimension
     * \param warpCol First Col of B to load
     *
     */
    __device__ void warpTileLoadB(SharedSize* bSharedAddr, const int kslice, const int kStride,
                                  const int warpCol) {
        // Page fragment into B.
        Coordinate warpBase{0, warpCol};
        for (int i = 0; i < numBFragments; i++) {
            // Determine which col we will load
            int fragmentRow = GetBaseFragmentCoordinate(warpBase, 0, i).col;
            int linearizedIndex =
                computeSwizzledIndex(kslice, kStride, fragmentRow, Mma::dims.n, computeBDimChunk);

            Mma::loadBMatrix_16_8(&bSharedAddr[linearizedIndex], B[i]);
        }
    }

    /** Compute linearized index of output fragment D>
     *
     * \param aFragIndex Index of A fragment of WarpTile ex. (0..3)
     * \param bFragIndex Index of B fragment of WarpTile ex. (0..7)
     *
     * \return Row-major linearized index of D fragment of WarpTile
     */
    __device__ constexpr int GetDIndex(const int aFragIndex, const int bFragIndex) {
        return aFragIndex * numBFragments + bFragIndex;
    }

    /** Compute A*B+D=D for a single k-slice of all fragments in a WarpTile. Accumulates in
     * place.
     *
     *
     */
    __device__ void warpTileMma() {
        for (int a = 0; a < numAFragments; a++) {
            for (int b = 0; b < numBFragments; b++) {
                Mma::FragmentD_16x8& Dfrag = D[GetDIndex(a, b)];
                Mma::mma_16_8_16(A[a], B[b], Dfrag, Dfrag);
            }
        }
    }

    /** Compute upper left coordinate of a single output fragment D of a WarpTile.
     *
     * \param warpBaseCoord - Upper left coordinate of the WarpTile.
     * \param aFragIndex A fragment used to compute the desired D fragment.
     * \param bFragIndex B fragment used to compute the desired D fragment.
     *
     * \return The upper left coordinate of the specified output fragment D.
     */
    __device__ Coordinate GetBaseFragmentCoordinate(const Coordinate& warpBaseCoord,
                                                    const int aFragIndex, const int bFragIndex) {
        int fragRow = warpBaseCoord.row + (aFragIndex * Mma::dims.m);
        int fragCol = warpBaseCoord.col + (bFragIndex * Mma::dims.n);
        return {fragRow, fragCol};
    }

    /** Final checks after MMA has completed. Iterate over all output elements of WarpTile and
     * compute distance.
     *
     * \param blockBaseCoord Upper left global coordinate of the block. Useful for shared memory
     * 	accesses since they are all relative to this.
     * \param warpBaseCoord Upper left global coordinate of WarpTile.
     * \param epsilonSqd The maximum distance two points can be apart to be considered a pair.
     * \param squaredQueries Array of sums of squared query points for the warp.
     * \param squaredCandidates Array of sums of squared candidate points for the warp.
     * \param searchShape The actual size of the similarity search. Extra points are padded on outer
     * dimensions of the search.
     *
     */
    __device__ int inspectResults(const Coordinate& blockBaseCoord, const Coordinate& warpBaseCoord,
                                  const float* squaredQueries, const float* squaredCandidates,
                                  const float epsilonSqd, const Mma::mmaShape& searchShape,
                                  Pairs::Pairs& pairs) {
        int count = 0;
#pragma unroll
        for (int a = 0; a < numAFragments; a++) {
#pragma unroll
            for (int b = 0; b < numBFragments; b++) {
                Coordinate fragCoords = GetBaseFragmentCoordinate(warpBaseCoord, a, b);
                Mma::FragmentD_16x8& Dfrag = D[GetDIndex(a, b)];
#pragma unroll
                for (int d = 0; d < numRegistersD; d++) {
                    int threadInWarp = threadIdx.x % WARPSIZE;
                    Coordinate elemCoord =
                        Mma::GetDElementCoordinate_16_8_float(fragCoords, threadInWarp, d);

                    int relativeQueryIndex = elemCoord.row - blockBaseCoord.row;
                    int relativeCandidateIndex = elemCoord.col - blockBaseCoord.col;
                    float querySquared = squaredQueries[relativeQueryIndex];
                    float candidateSquared = squaredCandidates[relativeCandidateIndex];
                    float distanceSqd = querySquared + candidateSquared -
                                        (static_cast<float>(2.0) * Dfrag.Registers[d]);

                    if (Debug) {
                        printf("OElement %d, %d q^2=%f c^2=%f d=%f d^2=%f\n", elemCoord.row,
                               elemCoord.col, querySquared, candidateSquared, Dfrag.Registers[d],
                               distanceSqd);
                    }

                    if (distanceSqd < epsilonSqd) {
                        // Ignore 0 padded points on boundaries
                        if (elemCoord.row < searchShape.m && elemCoord.col < searchShape.n) {
                            count++;
                            // Page the coordinates up
                            Pairs::Pair* pair = pairs.getSpace(1);
                            if (pair) {
                                pair->QueryPoint = elemCoord.row;
                                pair->CandidatePoint = elemCoord.col;
                            }
                        }
                    }
                }
            }
        }
        return count;
    }
};
};  // namespace WarpMma
#endif /* end of include guard: WARPMMA_CUH_OL9KOX7Y */
