#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <iostream>
#include <vector>

#include "blockMma.cuh"
#include "utils.cuh"

using SharedSize = WarpMma::SharedSize;
using InPrec = Mma::InPrec;

__global__ void MmaPtxShared(unsigned long long* iterationCount, SharedSize* AValues,
                             SharedSize* BValues, int kStride);
__device__ uint get_smid(void);

// ---------- Matrix Parameters ----------
constexpr int numPoints = 1024 * 16;
constexpr Mma::mmaShape globalMmaShape{numPoints, numPoints, 64 * 64};

// ---------- Mma parameters ----------

// ---------- Warp parameters ----------

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
    int aSize = sizeof(InPrec) * globalMmaShape.m * globalMmaShape.k;
    int bSize = sizeof(InPrec) * globalMmaShape.n * globalMmaShape.k;
    cudaMalloc(&d_AValues, aSize);
    cudaMalloc(&d_BValues, bSize);

    // Kind of a hack but we go to NaN if we let it keep incrementing
    int maxFloat = 32768;
    std::vector<half2> h_AValues{};
    // Fill the vector with increasing half-precision values
    // Note that this gets funny > 2048 because of imprecision of half values
    for (int m = 0; m < globalMmaShape.m; m++) {
        for (int k = 0; k < globalMmaShape.k; k += 2) {
            half2 val{};
            val.x = static_cast<half>(min(maxFloat, m * globalMmaShape.k + k));
            val.y = static_cast<half>(min(maxFloat, m * globalMmaShape.k + k + 1));
            h_AValues.push_back(val);
        }
    }

    PrintMatrix("Global A", reinterpret_cast<half*>(h_AValues.data()), globalMmaShape.m,
                globalMmaShape.k);

    std::vector<half2> h_BValues{};
    // Create identity matrix
    for (int row = 0; row < globalMmaShape.n; row++) {
        for (int col = 0; col < globalMmaShape.k; col += 2) {
            half2 val{0, 0};
            if (col == row)
                val.x = 1;
            else if (col + 1 == row)
                val.y = 1;
            h_BValues.push_back(val);
        }
    }
    PrintMatrix("Global B", reinterpret_cast<half*>(h_BValues.data()), globalMmaShape.n,
                globalMmaShape.k);

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

    dim3 gridDim(ceil(1.0 * globalMmaShape.n / BlockMma::GetBlockTileDims().n),
                 ceil(1.0 * globalMmaShape.m / BlockMma::GetBlockTileDims().m), 1);
    dim3 blockDim(BlockMma::numWarps * WARPSIZE, 1, 1);
    size_t sharedMemBytes = BlockMma::pipelineDepth * BlockMma::ElemsPerStage * sizeof(SharedSize);
    printf("Requesting %lu bytes of shared memory\n", sharedMemBytes);
    gpuErrchk(cudaFuncSetAttribute(MmaPtxShared, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   sharedMemBytes));
    MmaPtxShared<<<gridDim, blockDim, sharedMemBytes>>>(d_iterationCount, d_AValues, d_BValues,
                                                        globalMmaShape.k);

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
    const float tflops = static_cast<float>(globalMmaShape.m) * globalMmaShape.n *
                         globalMmaShape.k * 2 / elapsedTime / 1e12;
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
    BlockMma::BlockTileMma(iterationCount, AValues, BValues, kStride);
}
