#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#include <iostream>

#define WARPSIZE 32

// Low level namespace for fragments, and single mma operations
namespace Mma {
constexpr bool Debug = false;

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
using FragmentD_16x8 = Fragment<float, 4>;

__device__ inline uint32_t cvta_to_shared_u32(const void* pointer) {
    uint32_t address;
    // This is converting our generic 64 bit address to the shared memory state space. This
    // means subtracting the base address of the shared space, and then truncating to 32 bits
    // since shared memory all fits into 32 bits this is safe. The generic address space is 64
    // bits though.
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

// Load row of shared memory into Fragment
// Since this has inline PTX it only works for a specific template specification
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

// Load row of shared memory into Fragment
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

// TODO, this feels like I'm hard coding this...maybe it can be templated?
// Compute the coordinate of a specific register
__device__ Coordinate GetDElementCoordinate_16_8_float(Coordinate& baseCoord, int threadInWarp,
                                                       int dIndex) {
    int row, col;
    // First 8x8 matrix
    if (dIndex < 2) {
        row = baseCoord.row + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + dIndex;
    }
    // second 8x8 matrix
    else {
        row = baseCoord.row + 8 + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + (dIndex / 2);
    }
    return {row, col};
}

};  // namespace Mma

// A single warp operation made up of many fragments of A, B, and D
namespace WarpMma {

// Warp  Parameters
constexpr int numAFragments = 4;
constexpr int numBFragments = 8;
constexpr int numDFragments = numAFragments * numBFragments;

struct WarpTileDims {
    int m{};
    int n{};
    int k{};
};

// TODO Rounding might not work out here, have to be careful
constexpr int numRegistersA =
    Mma::dims.m * Mma::dims.k * sizeof(half) / sizeof(uint32_t) / WARPSIZE;
constexpr int numRegistersB =
    Mma::dims.k * Mma::dims.n * sizeof(half) / sizeof(uint32_t) / WARPSIZE;
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
// Template can specify the size of the actual warp tile.
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

    // TODO pass in k index here
    //  Given a WarpTile, loads all the A fragments into it
    __device__ void warpTileLoadA(half2* aTileAddr, Mma::Coordinate& baseCoord) {
        // Page fragment into A. This is 2 fragments
        for (int i = 0; i < numAFragments; i++) {
            // Upper left row
            int rowIndex = baseCoord.row + (i * Mma::dims.m);
            // TODO compute address for this thread
            Mma::loadAMatrix_16_16(aTileAddr, A[i]);
        }
    }

    // Given a WarpTile, loads all the B fragments into it
    __device__ void warpTileLoadB(half2* bTileAddr, Mma::Coordinate& baseCoord) {
        // Page fragment into B. This is 4 fragments
        // Need to duplicate addresses here for threads 16-31
        for (int i = 0; i < numBFragments; i++) {
            int colIndex = baseCoord.col + (i * Mma::dims.n);
            // TODO compute address for this thread
            Mma::loadBMatrix_16_8(bTileAddr, B[i]);
        }
    }

    __device__ int GetDIndex(const int aFragIndex, const int bFragIndex) {
        return aFragIndex * numBFragments + bFragIndex;
    }

    // Given a WarpTile with operands A and B, computes D=A*B+D
    __device__ void warpTileMma() {
        for (int a = 0; a < numAFragments; a++) {
            for (int b = 0; b < numBFragments; b++) {
                Mma::FragmentD_16x8& Dfrag = D[GetDIndex(a, b)];
                Mma::mma_16_8_16(A[a], B[b], Dfrag, Dfrag);
            }
        }
    }

    __device__ Mma::Coordinate GetBaseFragmentCoordinate(Mma::Coordinate& baseCoord,
                                                         const int aFragIndex,
                                                         const int bFragIndex) {
        int fragRow = baseCoord.row + (aFragIndex * Mma::dims.m);
        int fragCol = baseCoord.col + (bFragIndex * Mma::dims.n);
        return {fragRow, fragCol};
    }

    // Once computation is done, iterate over all output elements and apply epilogue
    __device__ void inspectResults(Mma::Coordinate& baseCoord) {
        for (int a = 0; a < numAFragments; a++) {
            for (int b = 0; b < numBFragments; b++) {
                Mma::Coordinate fragCoords = GetBaseFragmentCoordinate(baseCoord, a, b);
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
// Divide by two since it's an array of half2 values
constexpr BlockMma::BlockTileDims blockTileDims = GetBlockTileDims();
constexpr int aBlockTileSize = blockTileDims.m * blockTileDims.k / 2;
constexpr int bBlockTileSize = blockTileDims.n * blockTileDims.k / 2;

__device__ Mma::Coordinate GetBaseBlockCoordinate() {
    int baseRow = blockIdx.y * GetBlockTileDims().m;
    int baseCol = blockIdx.x * GetBlockTileDims().n;
    return {baseRow, baseCol};
}

__device__ Mma::Coordinate GetBaseWarpCoordinate(Mma::Coordinate baseBlockCoord, int warpId) {
    int warpRow = warpId / numWarpCols;
    int warpCol = warpId % numWarpCols;
    int baseRow = baseBlockCoord.row + warpRow * WarpMma::GetWarpTileDims().m;
    int baseCol = baseBlockCoord.col + warpCol * WarpMma::GetWarpTileDims().n;
    return {baseRow, baseCol};
}

__device__ void Mma(unsigned long long* iterationCount) {
    int tidx = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    // Kind of a useless count to get compiler to not optimize away my code
    unsigned int count = 0;

    // We need 16 byte alignment here since LDMatrix will read rows of 16B at a time
    // Are static shared memory allocations already aligned?
    __shared__ __align__(16) half2 ATile[aBlockTileSize];
    __shared__ __align__(16) half2 BTile[bBlockTileSize];

    // Page Global --> Shared Memory

    // Compute Upper left coordinate that this block is responsible for
    Mma::Coordinate baseBlockCoord = GetBaseBlockCoordinate();
    // Compute the Upper left coordinate that each warp is responsible for
    Mma::Coordinate baseWarpCoord = GetBaseWarpCoordinate(baseBlockCoord, warpId);

    // At this point we are in the kernel, so cuda parallelism is taking place. We really have
    // numWarps running
    WarpMma::WarpTile warpTile;
    warpTile.clearD();

    // Accumulate into D as many times as we need to
    for (int kslice = 0; kslice < BlockMma::kSlices; kslice++) {
        warpTile.warpTileLoadA(ATile, baseWarpCoord);
        warpTile.warpTileLoadB(BTile, baseWarpCoord);
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

__global__ void MmaPtxShared(unsigned long long* iterationCount);
__device__ uint get_smid(void);

constexpr bool Debug = false;

// ---------- Matrix Parameters ----------
constexpr int m = 1024;
constexpr int n = 1024;
constexpr int k = 1024;

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
    MmaPtxShared<<<gridDim, blockDim>>>(d_iterationCount);

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
}

__global__ void MmaPtxShared(unsigned long long* iterationCount) { BlockMma::Mma(iterationCount); }
