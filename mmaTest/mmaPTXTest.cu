#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#include <iostream>

namespace WarpMma {

// Represents one tile of matrix that ldmatrix loads into registers. Each thread holds a piece of
// this fragment. Input operands in half precision are packed in pairs into registers. Precision and
// fragment size dictate how many registers are required
template <typename T, int NumReg>
struct Fragment {
    T Reg[NumReg]{};
};

constexpr int numRegistersA = 4;
constexpr int numRegistersB = 2;
constexpr int numRegistersD = 4;
constexpr int numAFragments = 4;
constexpr int numBFragments = 8;
constexpr int numDFragments = 32;
// Each WarpTile stores multiple operands A and B to compute a large output matrix D
struct WarpTile {
    // 2 Fragments of A (16x16 values each) really half2, not uint32
    Fragment<uint32_t, numRegistersA> A[numAFragments]{};
    // 4 Fragments of B (16x8 values each) really half2, not uint32
    Fragment<uint32_t, numRegistersB> B[numBFragments]{};
    // 8 Fragments of D (16x8 values each)
    Fragment<float, numRegistersD> D[numDFragments]{};
};

template <int slices>
struct BlockTileLayout {
    int kSlices = slices;
};

};  // namespace WarpMma

__global__ void MmaPtxShared(unsigned long long* iterationCount);
__device__ uint get_smid(void);
__device__ void loadAMatrix_16_16(const void* smem_row_start, uint32_t* A);
__device__ void loadBMatrix_16_8(const void* smem_row_start, uint32_t* B);
__device__ void mma_16_8_16(const uint32_t* A, const uint32_t* B, const float* C, float* D);
__device__ void clearD(struct WarpMma::WarpTile& tile);
__device__ void warpTileMma(struct WarpMma::WarpTile& tile);
__device__ void warpTileLoadB(struct WarpMma::WarpTile& tile, half2* bTileAddr);
__device__ void warpTileLoadA(struct WarpMma::WarpTile& tile, half2* aTileAddr);

constexpr bool Debug = false;
constexpr long numIterations = 1e7;
constexpr int m = 16;
constexpr int n = 8;
constexpr int k = 16;
// Have a 128x128 block tile
constexpr int blockTileSize = 128;
// How many chunks of k to page into shared memory at at time
constexpr int kSlices = 4;
// Divide by two since it's an array of half2 values
constexpr int aBlockTileSize = blockTileSize * k * kSlices / 2;
constexpr int bBlockTileSize = aBlockTileSize;

constexpr int totalFlopsPerOp = m * n * k * 2;

constexpr int tensorCoresPerSM = 4;
constexpr int numSMs = 108;
constexpr int numWarpCols = 2;
constexpr int numWarpRows = 2;
constexpr int numWarps = numWarpCols * numWarpRows;
constexpr int numWaves = 1;

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

    dim3 gridDim(numWaves * numSMs, 1, 1);
    // 16 warps is the minimum to achieve 100% tensor core usage.
    // Interestingly, performance drops when you do 17 warps, likely because there are 4 tensor
    // cores, so we want a multiple of 4 warps for optimal performance.
    dim3 blockDim(32 * numWarps, 1, 1);
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
    const float tflops = gridDim.x * blockDim.x / 32.0 * numIterations * WarpMma::numDFragments *
                         totalFlopsPerOp / elapsedTime / 1e12;
    printf("Estimated TFLOPS %.3f\n", tflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_iterationCount);
}

__global__ void MmaPtxShared(unsigned long long* iterationCount) {
    int tidx = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    // Kind of a useless count to get compiler to not optimize away my code
    unsigned int count = 0;

    if (Debug) {
        if (tidx == 0) {
            printf("Block Index %d, %d running on SM %d\n", blockIdx.x, blockIdx.y, get_smid());
        }
    }

    // We need 16 byte alignment here since LDMatrix will read rows of 16B at a time
    // Are static shared memory allocations already aligned?
    __shared__ __align__(16) half2 ATile[aBlockTileSize];
    __shared__ __align__(16) half2 BTile[bBlockTileSize];

    if (Debug) {
        // Check to make sure the address it allocated is 16B aligned
        if (tidx == 0) {
            // I essentially get 0x0 for this value
            printf("ATile Address Alignment Check: %p\n", ATile);
            // With ATile being 128 half2 (2x2 bytes) I get 512 as the offset so this adds up
            printf("BTile Address Alignment Check: %p\n", BTile);
        }
    }

    // Have each thread copy one row of data into shared memory tile for A
    // Each thread copies 8 half values over
    // We have 4 fragments for A, so all threads participate
    for (int col = 0; col < 4; col++) {
        ATile[tidx * 4 + col].x = static_cast<half>(tidx * 8 + col * 2);
        ATile[tidx * 4 + col].y = static_cast<half>(tidx * 8 + col * 2 + 1);
    }

    // Have each thread copy one row fo data into shared memory tile for B
    // We have 2 fragments for B, so only half the threads participate
    if (tidx < 16) {
        for (int row = 0; row < 4; row++) {
            BTile[tidx * 4 + row].x = static_cast<half>(tidx * 8 + row * 2);
            BTile[tidx * 4 + row].y = static_cast<half>(tidx * 8 + row * 2 + 1);
        }
    }

    // Declare two warp tiles for double buffering. One tile is being transferred while the
    // other is being used by the mma instruction
    WarpMma::WarpTile warpTile;
    clearD(warpTile);

    // Repeat just the load matrix and calculation part of loop for benchmarking
    // This assumes that we are able to page from global memory to shared memory efficiently
    for (long i = 0; i < numIterations; i++) {
        // Accumulate into D as many times as we need to
        for (int kslice = 0; kslice < kSlices; kslice++) {
            // Need to pass in here:
            // First A address (based on what warp I am, and what kslice
            // Which vertical slice this warp cares about
            int aTileVerticalSliceIndex = warpId / numWarpCols;
            int aTileVerticalSliceSize = k * kSlices * blockTileSize / numWarpRows;
            // half2* aAddr = &ATile[aTileVerticalSliceIndex * aTileVerticalSliceSize + ];
            //  Frst B address
            //  A stride to get to next row (num k slices * k)
            //  B stride

            warpTileLoadA(warpTile, &ATile[tidx * 4]);
            warpTileLoadB(warpTile, &BTile[(tidx % 16) * 4]);
            warpTileMma(warpTile);
        }
    }

    // This is all here so everything isn't optimized away
    for (int i = 0; i < WarpMma::numDFragments; i++) {
        if (tidx == 0 && (warpTile.D[i].Reg[0] > 10.0f)) {
            count++;
        }
    }

    if (tidx == 0) {
        atomicAdd(iterationCount, count);
    }
}

__device__ void loadAMatrix_16_16(const void* smem_row_start, uint32_t* A) {
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
        : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect A
        for (int i = 0; i < 4; i++) {
            half2* tempVal = reinterpret_cast<half2*>(&A[i]);
            printf("Thread %d, %d: A%d=%f, A%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

__device__ void loadBMatrix_16_8(const void* smem_row_start, uint32_t* B) {
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
        : "=r"(B[0]), "=r"(B[1])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect B
        for (int i = 0; i < 2; i++) {
            half2* tempVal = reinterpret_cast<half2*>(&B[i]);
            printf("Thread %d, %d: B%d=%f, B%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

__device__ void mma_16_8_16(const uint32_t* A, const uint32_t* B, const float* C, float* D) {
    // 16x8x8 TC Operation
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7}, "
        " { %8, %9 }, "
        " { %10, %11, %12, %13 };"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));

    if (Debug) {
        printf("Thread %d, %d: D0=%f, D1=%f, D2=%f, D3=%f\n", threadIdx.x, threadIdx.y, D[0], D[1],
               D[2], D[3]);
    }
}

__device__ void clearD(struct WarpMma::WarpTile& tile) {
    for (int i = 0; i < WarpMma::numDFragments; i++) {
        for (int j = 0; j < WarpMma::numRegistersD; j++) {
            tile.D[i].Reg[j] = 0.0f;
        }
    }
}

// Given a WarpTile, loads all the A fragments into it
__device__ void warpTileLoadA(struct WarpMma::WarpTile& tile, half2* aTileAddr) {
    // Page fragment into A. This is 2 fragments
    for (int i = 0; i < WarpMma::numAFragments; i++) {
        // TODO aTileAddr can't be constant here obviously
        loadAMatrix_16_16(aTileAddr, tile.A[i].Reg);
    }
}

// Given a WarpTile, loads all the B fragments into it
__device__ void warpTileLoadB(struct WarpMma::WarpTile& tile, half2* bTileAddr) {
    // Page fragment into B. This is 4 fragments
    // Need to duplicate addresses here for threads 16-31
    for (int i = 0; i < WarpMma::numBFragments; i++) {
        // TODO bTileAddr can't be constant here
        loadBMatrix_16_8(bTileAddr, tile.B[i].Reg);
    }
}

// Given a WarpTile with operands A and B, computes all the output tiles D
__device__ void warpTileMma(struct WarpMma::WarpTile& tile) {
    // Perform MMA. Compute 8 fragments
    // 0-1
    for (int i = 0; i < WarpMma::numAFragments; i++) {
        // 0-3
        for (int j = 0; j < WarpMma::numBFragments; j++) {
            int dIndex = i * WarpMma::numBFragments + j;
            float* D = tile.D[dIndex].Reg;
            mma_16_8_16(tile.A[i].Reg, tile.B[j].Reg, D, D);
        }
    }
}
