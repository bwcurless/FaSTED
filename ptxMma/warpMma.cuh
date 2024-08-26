/******************************************************************************
 * File:             warpMma.cuh
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      All the functionality to compute a single warp tile.
 *****************************************************************************/

#ifndef WARPMMA_CUH_OL9KOX7Y
#define WARPMMA_CUH_OL9KOX7Y

#include "ptxMma.cuh"

// A single warp operation made up of many fragments of A, B, and D
namespace WarpMma {
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

    /** Loads a WarpTile's A fragments from relevant parts of shared memory.
     *
     * \param aSharedAddr Start of shared mem. array
     * \param kslice k slice to accumulate ex. (0..3)
     * \param kStride Shared mem. k dimension
     * \param warpRow First row of A to load
     *
     */
    // TODO A and B should really follow the same functions
    __device__ void warpTileLoadA(SharedSize* aSharedAddr, int kslice, int kStride, int warpRow) {
        // Page fragment into A.
        int laneId = threadIdx.x % WARPSIZE;
        Mma::Coordinate warpBase{warpRow, 0};
        int kStrideInt4 = kStride / dimPerInt4;
        for (int i = 0; i < numAFragments; i++) {
            // Determine which row we will load
            int fragmentRow = GetBaseFragmentCoordinate(warpBase, i, 0).row;
            int threadRow = fragmentRow + (laneId % Mma::dims.m);

            // Determine which chunk of cols we will load
            // base slice offset plus additional offset for lanes 16-31
            int threadCol = kslice * Mma::dims.k / dimPerInt4 + (laneId < Mma::dims.m ? 0 : 1);
            int swizzleRow = laneId % swizzleFactor;
            int swizzledCol = threadCol ^ swizzleRow;

            int linearizedIndex = threadRow * kStrideInt4 + swizzledCol;
            Mma::loadAMatrix_16_16(&aSharedAddr[linearizedIndex], A[i]);
        }
    }

    // Same as warpTileLoadA, but for B
    __device__ void warpTileLoadB(SharedSize* bSharedAddr, int kslice, int kStride, int warpCol) {
        // Page fragment into B.
        // B is stored row-major in shared memory
        int laneId = threadIdx.x % WARPSIZE;
        Mma::Coordinate warpBase{0, warpCol};
        int kStrideInt4 = kStride / dimPerInt4;
        for (int i = 0; i < numBFragments; i++) {
            // Determine which col we will load
            int fragmentRow = GetBaseFragmentCoordinate(warpBase, 0, i).col;
            int threadRow = fragmentRow + (laneId % Mma::dims.n);

            // Determine which chunk of rows we will load
            // base slice offset plus additional offset for lanes 8-15
            // threads 16-31 need to have the same addresses as 0-16 for the B Matrices
            int threadCol =
                kslice * Mma::dims.k / dimPerInt4 + (((laneId % 16) < Mma::dims.n) ? 0 : 1);
            int swizzleRow = laneId % swizzleFactor;
            int swizzledCol = threadCol ^ swizzleRow;

            int linearizedIndex = threadRow * kStrideInt4 + swizzledCol;
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
    __device__ int GetDIndex(const int aFragIndex, const int bFragIndex) {
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
    __device__ Mma::Coordinate GetBaseFragmentCoordinate(Mma::Coordinate& warpBaseCoord,
                                                         const int aFragIndex,
                                                         const int bFragIndex) {
        int fragRow = warpBaseCoord.row + (aFragIndex * Mma::dims.m);
        int fragCol = warpBaseCoord.col + (bFragIndex * Mma::dims.n);
        return {fragRow, fragCol};
    }

    /** Final checks after MMA has completed. Iterate over all output elements of WarpTile and
     * compute distance.
     *
     * \param warpBaseCoord Upper left global coordinate of WarpTile.
     *
     */
    __device__ int inspectResults(Mma::Coordinate& warpBaseCoord, float epsilon) {
        int count = 0;
        for (int a = 0; a < numAFragments; a++) {
            for (int b = 0; b < numBFragments; b++) {
                Mma::Coordinate fragCoords = GetBaseFragmentCoordinate(warpBaseCoord, a, b);
                Mma::FragmentD_16x8& Dfrag = D[GetDIndex(a, b)];
                for (int d = 0; d < numRegistersD; d++) {
                    int threadInWarp = threadIdx.x % WARPSIZE;
                    Mma::Coordinate elemCoord =
                        Mma::GetDElementCoordinate_16_8_float(fragCoords, threadInWarp, d);
                    if (Debug) {
                        printf("OElement %d, %d = %f\n", elemCoord.row, elemCoord.col,
                               Dfrag.Registers[d]);
                    }
                    // TODO perform addition of squared terms and comparison with epsilon here
                    if (Dfrag.Registers[d] > epsilon) {
                        count++;
                    }
                }
            }
        }
        return count;
    }
};
};  // namespace WarpMma
#endif /* end of include guard: WARPMMA_CUH_OL9KOX7Y */
