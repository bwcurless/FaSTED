#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#include <iostream>
#include <vector>

#define WARPSIZE 32

__device__ __host__ constexpr bool IsDivisionExact(int dividend, int divisor) {
    return dividend == (divisor * (dividend / divisor));
}

// Compile time fails if constexpr division results in any rounding
template <size_t dividend, size_t divisor>
__device__ void Assert_Is_Division_Exact() {
    // Check if the division is exact
    // Forced round down will happen here due to truncation.
    static_assert(IsDivisionExact(dividend, divisor), "Division is not exact.");
}

// Low level namespace for fragments, and single mma operations
namespace Mma {
constexpr bool Debug = false;

using InPrec = half;
using OutPrec = float;

struct mmaTileDims {
    int m{};
    int n{};
    int k{};
};

struct Coordinate {
    int row{};
    int col{};
};

// Declare a constant for reference elsewhere
// Dimensions of a fundamental mma operation in ptx
constexpr mmaTileDims dims{16, 8, 16};

// Represents one operand A, B, C, or D, of an mma operaetion that ldmatrix loads into registers.
// Each thread holds a piece of this operand. Input operands in half precision are packed in pairs
// into registers. Precision and fragment size dictate how many registers are required
template <typename T, int NumReg>
struct Fragment {
   public:
    T Registers[NumReg]{};

    __device__ void clear() {
        for (int j = 0; j < NumReg; j++) {
            Registers[j] = 0.0f;
        }
    }
};

// Making some aliases to make it easier to define specific functions taking only these template
// arguments due to the fact that they require inline ptx and aren't generic
using FragmentA_16x16 = Fragment<uint32_t, 4>;
using FragmentB_16x8 = Fragment<uint32_t, 2>;
using FragmentD_16x8 = Fragment<OutPrec, 4>;

/** Convert pointer to shared memory state space. This is converting our generic 64 bit address to
 * the shared memory state space. This means subtracting the base address of the shared space, and
 * then truncating to 32 bits. Since shared memory all fits into 32 bits this is safe. The generic
 * address space is 64 bits.
 *
 * \param pointer The generic pointer to convert
 *
 * \return 32-bit pointer to shared memory
 */
__device__ uint32_t cvta_to_shared_u32(const void* pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

/** Load row of shared memory into 16x16 Fragment.
 *
 *
 *
 * \param smem_row_start int4* to shared memory
 * \param A fragment to load into
 *
 */
__device__ void loadAMatrix_16_16(const void* smem_row_start, FragmentA_16x16& A) {
    //  Page into A
    //  Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(smem_row_start);

    if (Debug) {
        if (threadIdx.x == 0) {
            printf("Shared 32b Memory Address A 0x%x\n", smem_ptr);
        }
    }

    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=r"(A.Registers[0]), "=r"(A.Registers[1]), "=r"(A.Registers[2]), "=r"(A.Registers[3])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect A
        for (int i = 0; i < 4; i++) {
            half2* tempVal = reinterpret_cast<half2*>(A.Registers[i]);
            printf("Thread %d, %d: A%d=%f, A%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

/** Load row of shared memory into 16x8 Fragment.
 *
 *
 *
 * \param smem_row_start int4* to shared memory
 * \param B fragment to load into
 *
 */
__device__ void loadBMatrix_16_8(const void* smem_row_start, FragmentB_16x8& B) {
    //  Page into A
    //  Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(smem_row_start);

    if (Debug) {
        if (threadIdx.x == 0) {
            printf("Shared 32b Memory Address B 0x%x\n", smem_ptr);
        }
    }

    // To get this to work out like the example, you don't want to transpose here.
    // It seems there is an implied transpose for the B matrix when you execute mma
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
        "{ %0, %1 }, [%2];"
        : "=r"(B.Registers[0]), "=r"(B.Registers[1])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect B
        for (int i = 0; i < 2; i++) {
            half2* tempVal = reinterpret_cast<half2*>(&B.Registers[i]);
            printf("Thread %d, %d: B%d=%f, B%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

/** Computes A*B+C=D
 *
 *
 *
 * \param A Input fragment operand
 * \param B Input fragment operand
 * \param C Input fragment operand
 * \param D Output fragment operand
 *
 */
__device__ void mma_16_8_16(const FragmentA_16x16& A, const FragmentB_16x8& B,
                            const FragmentD_16x8& C, FragmentD_16x8& D) {
    // 16x8x8 TC Operation
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7}, "
        " { %8, %9 }, "
        " { %10, %11, %12, %13 };"
        : "=f"(D.Registers[0]), "=f"(D.Registers[1]), "=f"(D.Registers[2]), "=f"(D.Registers[3])
        : "r"(A.Registers[0]), "r"(A.Registers[1]), "r"(A.Registers[2]), "r"(A.Registers[3]),
          "r"(B.Registers[0]), "r"(B.Registers[1]), "f"(C.Registers[0]), "f"(C.Registers[1]),
          "f"(C.Registers[2]), "f"(C.Registers[3]));

    if (Debug) {
        printf("Thread %d, %d: D0=%f, D1=%f, D2=%f, D3=%f\n", threadIdx.x, threadIdx.y,
               D.Registers[0], D.Registers[1], D.Registers[2], D.Registers[3]);
    }
}

/** Compute the global coordinates of a specific output register.
 *
 *
 *
 * \param baseCoord Upper left global coordinate of a single Mma operation's output matrix D
 * \param threadInWarp Warp lane (0..31)
 * \param dIndex Index of output element (0..4)
 *
 * \return Global coordinates of this element
 */
__device__ Coordinate GetDElementCoordinate_16_8_float(Coordinate& baseCoord, int threadInWarp,
                                                       int dIndex) {
    // TODO, this feels like I'm hard coding this...maybe it can be templated?
    int row, col;
    // First 8x8 matrix
    if (dIndex < 2) {
        row = baseCoord.row + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + dIndex;
    }
    // second 8x8 matrix
    else {
        row = baseCoord.row + 8 + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + (dIndex % 2);
    }
    return {row, col};
}
};  // namespace Mma

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
     *
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
            int threadCol = kslice * Mma::dims.k / dimPerInt4 + (laneId > Mma::dims.m ? 1 : 0);
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
                kslice * Mma::dims.k / dimPerInt4 + (((laneId % 16) < Mma::dims.n) ? 1 : 0);
            int swizzleRow = laneId % swizzleFactor;
            int swizzledCol = threadCol ^ swizzleRow;

            int linearizedIndex = threadRow * kStrideInt4 + swizzledCol;
            Mma::loadBMatrix_16_8(&bSharedAddr[linearizedIndex], B[i]);
        }
    }

    /** Compute linearized index of output fragment D>
     *
     *
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

    /** Final checks after MMA has completed.
     *
     * Iterate over all output elements of WarpTile and compute distance.
     *
     * \param warpBaseCoord Upper left global coordinate of WarpTile.
     *
     */
    __device__ void inspectResults(Mma::Coordinate& warpBaseCoord) {
        for (int a = 0; a < numAFragments; a++) {
            for (int b = 0; b < numBFragments; b++) {
                Mma::Coordinate fragCoords = GetBaseFragmentCoordinate(warpBaseCoord, a, b);
                Mma::FragmentD_16x8& Dfrag = D[GetDIndex(a, b)];
                for (int d = 0; d < numRegistersD; d++) {
                    int threadInWarp = threadIdx.x % WARPSIZE;
                    Mma::Coordinate elemCoord =
                        Mma::GetDElementCoordinate_16_8_float(fragCoords, threadInWarp, d);
                    // TODO perform addition of squared terms and comparison with epsilon here
                }
            }
        }
    }
};
};  // namespace WarpMma

namespace BlockMma {

// Block Parameters
constexpr int numWarpCols = 2;
constexpr int numWarpRows = 2;
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
 *
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

/** Computes an mma operation at the block scope.
 *
 * \param iterationCount TODO Delete this
 * \param AValues Global memory array containing elements of A
 * \param BValues Global memory array containing elements of B
 * \param globalKStride Global k dimension of MMA
 *
 */
__device__ void Mma(unsigned long long* iterationCount, SharedSize* AValues, SharedSize* BValues,
                    int globalKStride) {
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

    // All threads colloborate to move Global-->Shared as data in shared is shared between all warps
    // in the block
    // Page A in
    // Each thread does an int4 copy from global to shared
    // Each warp would be copying 4 points over of 64 D each
    // TODO make this work for multiple K iterations
    // How many vectorized loads are required per point
    constexpr int copiesPerPoint = GetBlockTileDims().k / WarpMma::dimPerInt4;  // 16
    int copyLane = threadIdx.x % copiesPerPoint;
    // First 16 threads copy over point 1, next 16 point 2, so on until all points are satisfied
    for (int blockQueryIndex = threadIdx.x / copiesPerPoint; blockQueryIndex < GetBlockTileDims().m;
         blockQueryIndex += blockDim.x / copiesPerPoint) {
        int globalQueryIndex = blockQueryIndex + baseBlockCoord.row;
        SharedSize values =
            AValues[(globalQueryIndex * globalKStride / WarpMma::dimPerInt4) + copyLane];

        // Compute swizzled shared mem location
        int swizzledLane = copyLane ^ WarpMma::swizzleFactor;
        int swizzledAddress = blockQueryIndex * GetBlockTileDims().k + swizzledLane;

        ATile[swizzledAddress] = values;
    }

    // Page B in

    // Create numWarps warpTiles to process what this block is responsible for
    WarpMma::WarpTile warpTile;
    warpTile.clearD();

    // Accumulate into D as many times as we need to
    for (int kslice = 0; kslice < BlockMma::kSlices; kslice++) {
        warpTile.warpTileLoadA(ATile, kslice, blockTileDims.k, baseLocalWarpCoord.row);
        warpTile.warpTileLoadB(BTile, kslice, blockTileDims.k, baseLocalWarpCoord.col);
        warpTile.warpTileMma();
    }

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

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

__global__ void MmaPtxShared(unsigned long long* iterationCount, SharedSize* AValues,
                             SharedSize* BValues, int kStride);
__device__ uint get_smid(void);

constexpr bool Debug = false;

// ---------- Matrix Parameters ----------
constexpr int m = 128;
constexpr int n = 128;
constexpr int k = 64;

// ---------- Mma parameters ----------
constexpr int totalFlopsPerOp = Mma::dims.m * Mma::dims.n * Mma::dims.k * 2;

// ---------- Warp parameters ----------

// ---------- Block parameters ----------
// How many warps to launch per block
constexpr int numWarps = BlockMma::numWarpCols * BlockMma::numWarpRows;

// ---------- Hardware parameters ----------

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Return the ID of the steaming multiprocesser this block is running on
__device__ uint get_smid(void) {
    uint ret;

    asm("mov.u32 %0, %smid;" : "=r"(ret));

    return ret;
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

    SharedSize *d_AValues, *d_BValues;
    int aSize = sizeof(InPrec) * m * k;
    int bSize = sizeof(InPrec) * n * k;
    cudaMalloc(&d_AValues, aSize);
    cudaMalloc(&d_BValues, bSize);

    std::vector<half2> h_AValues{};
    // Fill the vector with increasing half-precision values
    for (int i = 0; i <= m * k / 2; i += 2) {
        half2 val{};
        val.x = i;
        val.y = i + 1;
        h_AValues.push_back(val);
    }

    std::vector<half2> h_BValues{};
    // Create identity matrix
    for (int row = 0; row < k; row++) {
        for (int col = 0; col < n / 2; col += 2) {
            half2 val{};
            val.x = 0;
            val.y = 0;
            if (col == row)
                val.x = 1;
            else if (col + 1 == row)
                val.y = 1;
            h_BValues.push_back(val);
        }
    }

    cudaMemcpy(d_AValues, h_AValues.data(), aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BValues, h_BValues.data(), bSize, cudaMemcpyHostToDevice);

    printf("Running kernel\n");

    // Each MMA operation is a 16x8x16 operation
    // There are 16x8 values calculated per warp of 32 threads
    // Each value requires 2 * 16 FLOPS due to the multiply and add
    // Total flops per warp per iteration is (16*8)*(2*16)=4096
    // Theoretical max for A100 is 312 TFLOPS
    // We have 4 Tensor cores per SM, 108 SM's, so 432 total TC's/GPU
    // This means each tensor core can do 312 TFLOPS / 432 TC's = 722 GFLOPS
    // If each operation is 4096 FLOP, then we would expect 176M mma operations per second
    cudaEventRecord(start, 0);

    dim3 gridDim(ceil(1.0 * n / BlockMma::GetBlockTileDims().n),
                 ceil(1.0 * m / BlockMma::GetBlockTileDims().m), 1);
    // 16 warps is the minimum to achieve 100% tensor core usage.
    // Interestingly, performance drops when you do 17 warps, likely because there are 4 tensor
    // cores, so we want a multiple of 4 warps for optimal performance.
    dim3 blockDim(WARPSIZE * numWarps, 1, 1);
    MmaPtxShared<<<gridDim, blockDim>>>(d_iterationCount, d_AValues, d_BValues, k);

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
    const float tflops = gridDim.x * blockDim.x / 32.0 * BlockMma::coarseFactor *
                         BlockMma::kSlices * WarpMma::numDFragments * totalFlopsPerOp /
                         elapsedTime / 1e12;
    printf("Estimated TFLOPS %.3f\n", tflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_iterationCount);
    cudaFree(d_AValues);
    cudaFree(d_BValues);
}

__global__ void MmaPtxShared(unsigned long long* iterationCount, SharedSize* AValues,
                             SharedSize* BValues, int kStride) {
    BlockMma::Mma(iterationCount, AValues, BValues, kStride);
}
