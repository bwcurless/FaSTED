#include <cuda.h>
#include <cuda_fp16.h>
#include <vector_types.h>

#include <iostream>

__global__ void MmaPtxShared();

int main(int argc, char* argv[]) {
    // Launch MMA Kernel
    dim3 gridDim(1, 1, 0);
    dim3 blockDim(32, 1, 0);
    MmaPtxShared<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
}

__global__ void MmaPtxShared() {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;

    // Each thread in has 4 32 bit registers for A
    // Each thread has 2 32 bit registers for B
    // Each thread has 4 floats for outputs
    float D[4];
    uint32_t A[4];
    uint32_t B[2];

    // We need 16 byte alignment here since LDMatrix will read rows of 16B at a time
    __shared__ __align__(16) half2 ATile[128];
    __shared__ __align__(16) half2 BTile[64];

    // Check to make sure the address it allocated is 16B aligned
    if (tidx == 0) {
        printf("ATile Address Alignment Check: %p\n", ATile);
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

    smem_ptr = reinterpret_cast<uint32_t>(&ATile[tidx * 4]);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
        : "r"(smem_ptr));

    // Inspect A
    /*
    for (int i = 0; i < 4; i++) {
        half2 tempVal = reinterpret_cast<half2>(A[i]);
        printf("Thread %d, %d: A%d=%f, A%d=%f", tidx, tidy, i, __half2float(tempVal.x), i,
               __half2float(tempVal.y));
    }
    */

    // Page into B
    // Need to duplicate addresses here for threads 16-31
    smem_ptr = reinterpret_cast<uint32_t>(&BTile[(tidx % 16) * 4]);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
        "{ %0, %1 }, [%2];"
        : "=r"(B[0]), "=r"(B[1])
        : "r"(smem_ptr));

    // Inspect B

    // Clear D
    D[0] = 0.0;
    D[1] = 0.0;

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
    printf("Thread %d, %d: D0=%f, D1=%f", tidx, tidy, D[0], D[1]);
}
