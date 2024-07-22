#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

#include <iostream>

__global__ void MmaPtxShared(unsigned long long* iterationCount);
__device__ uint get_smid(void);

constexpr bool Debug = false;
constexpr long numIterations = 1e7;
constexpr int m = 16;
constexpr int n = 8;
constexpr int k = 16;
constexpr int totalFlopsPerOp = m * n * k * 2;

constexpr int tensorCoresPerSM = 4;
constexpr int numSMs = 108;
constexpr int numWaves = 1;
constexpr int numTensorCores = numSMs * tensorCoresPerSM;

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
    dim3 blockDim(32 * 20, 1, 1);
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
    const float tflops =
        gridDim.x * blockDim.x / 32.0 * numIterations * totalFlopsPerOp / elapsedTime / 1e12;
    printf("Estimated TFLOPS %.3f\n", tflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_iterationCount);
}

__global__ void MmaPtxShared(unsigned long long* iterationCount) {
    int tidx = threadIdx.x % 32;
    int tidy = threadIdx.y;
    // Kind of a useless count to get compiler to not optimize away my code
    unsigned int count = 0;

    if (Debug) {
        if (tidx == 0 && tidy == 0) {
            printf("Block Index %d, %d running on SM %d\n", blockIdx.x, blockIdx.y, get_smid());
        }
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
    // Repeat just the load matrix and calculation part of loop for benchmarking
    // This assumes that we are able to page from global memory to shared memory efficiently
    constexpr int unrollFactor = 1;

    // Clear D
    D[0] = 0.0;
    D[1] = 0.0;
    D[2] = 0.0;
    D[3] = 0.0;

    //  Page into A
    //  Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(&ATile[tidx * 4]);

    if (Debug) {
        if (tidx == 0) {
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
            printf("Thread %d, %d: A%d=%f, A%d=%f\n", tidx, tidy, i, __half2float(tempVal->x), i,
                   __half2float(tempVal->y));
        }
    }

    // Page into B
    // Need to duplicate addresses here for threads 16-31
    smem_ptr = cvta_to_shared_u32(&BTile[(tidx % 16) * 4]);

    if (Debug) {
        if (tidx == 0) {
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
            printf("Thread %d, %d: B%d=%f, B%d=%f\n", tidx, tidy, i, __half2float(tempVal->x), i,
                   __half2float(tempVal->y));
        }
    }

    for (long i = 0; i < numIterations / unrollFactor; i++) {
        // for (int j = 0; j < unrollFactor; j++) {

        // Perform MMA
        // 16x8x8 TC Operation
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            " { %0, %1, %2, %3 }, "
            " { %4, %5, %6, %7}, "
            " { %8, %9 }, "
            " { %10, %11, %12, %13 };"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]),
              "f"(D[1]), "f"(D[2]), "f"(D[3]));

        // Inspect Results of D
        /* These are the expected resultant full matrices and fragments for A, B, and D
            These match the pictures in the Nvidia talk "Pushing tensor cores to the limit..."
        a0: [[ 0  1  2  3  4  5  6  7]
         [ 8  9 10 11 12 13 14 15]
         [16 17 18 19 20 21 22 23]
         [24 25 26 27 28 29 30 31]
         [32 33 34 35 36 37 38 39]
         [40 41 42 43 44 45 46 47]
         [48 49 50 51 52 53 54 55]
         [56 57 58 59 60 61 62 63]]
        a1: [[ 64  65  66  67  68  69  70  71]
         [ 72  73  74  75  76  77  78  79]
         [ 80  81  82  83  84  85  86  87]
         [ 88  89  90  91  92  93  94  95]
         [ 96  97  98  99 100 101 102 103]
         [104 105 106 107 108 109 110 111]
         [112 113 114 115 116 117 118 119]
         [120 121 122 123 124 125 126 127]]
        a2: [[128 129 130 131 132 133 134 135]
         [136 137 138 139 140 141 142 143]
         [144 145 146 147 148 149 150 151]
         [152 153 154 155 156 157 158 159]
         [160 161 162 163 164 165 166 167]
         [168 169 170 171 172 173 174 175]
         [176 177 178 179 180 181 182 183]
         [184 185 186 187 188 189 190 191]]
        a3: [[192 193 194 195 196 197 198 199]
         [200 201 202 203 204 205 206 207]
         [208 209 210 211 212 213 214 215]
         [216 217 218 219 220 221 222 223]
         [224 225 226 227 228 229 230 231]
         [232 233 234 235 236 237 238 239]
         [240 241 242 243 244 245 246 247]
         [248 249 250 251 252 253 254 255]]
           a_full:
        [[  0   1   2   3   4   5   6   7 128 129 130 131 132 133 134 135]
         [  8   9  10  11  12  13  14  15 136 137 138 139 140 141 142 143]
         [ 16  17  18  19  20  21  22  23 144 145 146 147 148 149 150 151]
         [ 24  25  26  27  28  29  30  31 152 153 154 155 156 157 158 159]
         [ 32  33  34  35  36  37  38  39 160 161 162 163 164 165 166 167]
         [ 40  41  42  43  44  45  46  47 168 169 170 171 172 173 174 175]
         [ 48  49  50  51  52  53  54  55 176 177 178 179 180 181 182 183]
         [ 56  57  58  59  60  61  62  63 184 185 186 187 188 189 190 191]
         [ 64  65  66  67  68  69  70  71 192 193 194 195 196 197 198 199]
         [ 72  73  74  75  76  77  78  79 200 201 202 203 204 205 206 207]
         [ 80  81  82  83  84  85  86  87 208 209 210 211 212 213 214 215]
         [ 88  89  90  91  92  93  94  95 216 217 218 219 220 221 222 223]
         [ 96  97  98  99 100 101 102 103 224 225 226 227 228 229 230 231]
         [104 105 106 107 108 109 110 111 232 233 234 235 236 237 238 239]
         [112 113 114 115 116 117 118 119 240 241 242 243 244 245 246 247]
         [120 121 122 123 124 125 126 127 248 249 250 251 252 253 254 255]]
        b0: [[ 0  8 16 24 32 40 48 56]
         [ 1  9 17 25 33 41 49 57]
         [ 2 10 18 26 34 42 50 58]
         [ 3 11 19 27 35 43 51 59]
         [ 4 12 20 28 36 44 52 60]
         [ 5 13 21 29 37 45 53 61]
         [ 6 14 22 30 38 46 54 62]
         [ 7 15 23 31 39 47 55 63]]
        b1: [[ 64  72  80  88  96 104 112 120]
         [ 65  73  81  89  97 105 113 121]
         [ 66  74  82  90  98 106 114 122]
         [ 67  75  83  91  99 107 115 123]
         [ 68  76  84  92 100 108 116 124]
         [ 69  77  85  93 101 109 117 125]
         [ 70  78  86  94 102 110 118 126]
         [ 71  79  87  95 103 111 119 127]]
        b_full:
        [[  0   8  16  24  32  40  48  56]
         [  1   9  17  25  33  41  49  57]
         [  2  10  18  26  34  42  50  58]
         [  3  11  19  27  35  43  51  59]
         [  4  12  20  28  36  44  52  60]
         [  5  13  21  29  37  45  53  61]
         [  6  14  22  30  38  46  54  62]
         [  7  15  23  31  39  47  55  63]
         [ 64  72  80  88  96 104 112 120]
         [ 65  73  81  89  97 105 113 121]
         [ 66  74  82  90  98 106 114 122]
         [ 67  75  83  91  99 107 115 123]
         [ 68  76  84  92 100 108 116 124]
         [ 69  77  85  93 101 109 117 125]
         [ 70  78  86  94 102 110 118 126]
         [ 71  79  87  95 103 111 119 127]]
        d:
        [[ 71192  79832  88472  97112 105752 114392 123032 131672]
         [ 75736  85400  95064 104728 114392 124056 133720 143384]
         [ 80280  90968 101656 112344 123032 133720 144408 155096]
         [ 84824  96536 108248 119960 131672 143384 155096 166808]
         [ 89368 102104 114840 127576 140312 153048 165784 178520]
         [ 93912 107672 121432 135192 148952 162712 176472 190232]
         [ 98456 113240 128024 142808 157592 172376 187160 201944]
         [103000 118808 134616 150424 166232 182040 197848 213656]
         [107544 124376 141208 158040 174872 191704 208536 225368]
         [112088 129944 147800 165656 183512 201368 219224 237080]
         [116632 135512 154392 173272 192152 211032 229912 248792]
         [121176 141080 160984 180888 200792 220696 240600 260504]
         [125720 146648 167576 188504 209432 230360 251288 272216]
         [130264 152216 174168 196120 218072 240024 261976 283928]
         [134808 157784 180760 203736 226712 249688 272664 295640]
         [139352 163352 187352 211352 235352 259352 283352 307352]]
        */

        if (Debug) {
            printf("Thread %d, %d: D0=%f, D1=%f, D2=%f, D3=%f\n", tidx, tidy, D[0], D[1], D[2],
                   D[3]);
        }
    }
    //    }
    if (tidx == 0 && (D[0] > 10.0f)) {
        count++;
    }

    // This is really just because I was doubting it was executing all this
    if (tidx == 0) {
        atomicAdd(iterationCount, count);
    }
}
