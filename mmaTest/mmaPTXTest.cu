#include <cuda.h>
#include <cuda_fp16.h>
#include <vector_types.h>

#include <iostream>

__global__ void MmaPtxShared();
__device__ uint get_smid(void);

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
    // This is converting our generic 64 bit address to the shared memory state space. This means
    // subtracting the base address of the shared space, and then truncating to 32 bits since shared
    // memory all fits into 32 bits this is safe. The generic address space is 64 bits though.
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
    // Launch MMA Kernel
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);
    printf("Running kernel\n");
    MmaPtxShared<<<gridDim, blockDim>>>();

    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void MmaPtxShared() {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    if (tidx == 0 && tidy == 0) {
        printf("Block Index %d, %d running on SM %d\n", blockIdx.x, blockIdx.y, get_smid());
    }

    // Each thread in has 4 32 bit registers for A
    // Each thread has 2 32 bit registers for B
    // Each thread has 4 floats for outputs
    float D[4];
    uint32_t A[4];
    uint32_t B[2];

    // We need 16 byte alignment here since LDMatrix will read rows of 16B at a time
    // Are static shared memory allocations already aligned?
    __shared__ __align__(16) half2 ATile[128];
    __shared__ __align__(16) half2 BTile[64];

    // Check to make sure the address it allocated is 16B aligned
    if (tidx == 0) {
        // I essentially get 0x0 for this value
        printf("ATile Address Alignment Check: %p\n", ATile);
        // With ATile being 128 half2 (2x2 bytes) I get 512 as the offset so this adds up
        printf("BTile Address Alignment Check: %p\n", BTile);
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

    // Page into A
    // Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(&ATile[tidx * 4]);

    if (tidx == 0) {
        printf("Shared 32b Memory Address A 0x%x\n", smem_ptr);
    }

    if (tidx == 0) {
        printf("Loading A Matrix\n");
    }

    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
        : "r"(smem_ptr));

    if (tidx == 0) {
        printf("Loading A Matrix done\n");
    }

    // Inspect A
    for (int i = 0; i < 4; i++) {
        half2* tempVal = reinterpret_cast<half2*>(&A[i]);
        printf("Thread %d, %d: A%d=%f, A%d=%f\n", tidx, tidy, i, __half2float(tempVal->x), i,
               __half2float(tempVal->y));
    }

    if (tidx == 0) {
        printf("Loading B Matrix\n");
    }

    // Page into B
    // Need to duplicate addresses here for threads 16-31
    smem_ptr = cvta_to_shared_u32(&BTile[(tidx % 16) * 4]);

    if (tidx == 0) {
        printf("Shared 32b Memory Address B 0x%x\n", smem_ptr);
    }

    // To get this to work out like the example, you don't want to transpose here.
    // It seems there is an implied transpose for the B matrix when you execute mma
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
        "{ %0, %1 }, [%2];"
        : "=r"(B[0]), "=r"(B[1])
        : "r"(smem_ptr));

    // Inspect B
    for (int i = 0; i < 2; i++) {
        half2* tempVal = reinterpret_cast<half2*>(&B[i]);
        printf("Thread %d, %d: B%d=%f, B%d=%f\n", tidx, tidy, i, __half2float(tempVal->x), i,
               __half2float(tempVal->y));
    }

    // Clear D
    D[0] = 0.0;
    D[1] = 0.0;
    D[2] = 0.0;
    D[3] = 0.0;

    // Perform MMA
    // 16x8x8 TC Operation
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7}, "
        " { %8, %9 }, "
        " { %10, %11, %12, %13 };"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]),
          "f"(D[2]), "f"(D[3]));

    // Inspect Results of D
    // Expected value for T0 D[0] is 71192. It is the dot product of
    // [0...7,64...71] and [0...7,128...135]
    printf("Thread %d, %d: D0=%f, D1=%f, D2=%f, D3=%f\n", tidx, tidy, D[0], D[1], D[2], D[3]);
}
