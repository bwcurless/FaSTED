#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <vector_types.h>

#include <cuda/pipeline>
#include <iostream>

__global__ void MatrixTransferKernel(int4* matrix, int kStride, unsigned long long* iterationCount);

constexpr bool Debug = false;

// Launch Params
constexpr int warpSize = 32;
constexpr int numWarps = 4;
constexpr int blockNumThreads = warpSize * numWarps;

// Block Params
constexpr int rowsPerBlock = 128;
constexpr int columnsPerStage = 64;
constexpr int totalStages = 64;
constexpr int elementsPerStage = columnsPerStage * rowsPerBlock;
constexpr int pipelineDepth = 2;
// How many half elements fit inside one int4 copy
constexpr int elementsPerInt4 = sizeof(int4) / sizeof(half);  // 8

// ---------- Matrix Parameters ----------
// Matrix to transfer is composed of half precision floats
constexpr int numRows = 1024 * 16;
constexpr int numColumns = columnsPerStage * totalStages;

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {
    cudaSetDevice(0);

    // Allocate this just so kernel actually does something
    unsigned long long* d_iterationCount;
    unsigned long long h_iterationCount = 0;
    gpuErrchk(cudaMalloc(&d_iterationCount, sizeof(unsigned long long)));
    gpuErrchk(cudaMemcpy(d_iterationCount, &h_iterationCount, sizeof(unsigned long long),
                         cudaMemcpyHostToDevice));

    int4* d_Matrix;
    size_t matrixSize = sizeof(half) * numRows * numColumns;
    gpuErrchk(cudaMalloc(&d_Matrix, matrixSize));

    dim3 gridDim(1, numRows / rowsPerBlock, 1);
    dim3 blockDim(blockNumThreads, 1, 1);
    size_t sharedMemBytes = pipelineDepth * elementsPerStage * sizeof(half);
    printf("Requesting %lu bytes of shared memory\n", sharedMemBytes);

    gpuErrchk(cudaFuncSetAttribute(MatrixTransferKernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes));
    MatrixTransferKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_Matrix, numColumns,
                                                                d_iterationCount);

    gpuErrchk(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(d_iterationCount);
    cudaFree(d_Matrix);
}

__device__ int SwizzleAddress(int sharedMemRow, int sharedMemColumn, int columnsPerRow) {
    // Column ^ Row in shared memory
    int swizzledLane = sharedMemColumn ^ (sharedMemRow % columnsPerRow);
    int swizzledAddress = sharedMemRow * columnsPerRow + swizzledLane;
    return swizzledAddress;
}

__device__ void LoadGlobalToSharedAsync(cuda::pipeline<cuda::thread_scope_thread>& pipe,
                                        int4* globalArray, int4* sharedArray, int firstGlobalRow,
                                        int numRows, int firstGlobalCol, int globalColumns) {
    // How many int4 values must be copied to copy an entire chunk of a row over
    constexpr int copiesPerStageRow = columnsPerStage / elementsPerInt4;  // 8

    int firstRow = threadIdx.x / copiesPerStageRow;
    int firstCol = threadIdx.x % copiesPerStageRow;
    int globalLeadingDim = globalColumns / elementsPerInt4;
    int rowStride = blockDim.x / copiesPerStageRow;  // Block copies 16 points/iteration

    // First 8 threads copy over point 1, next 8 point 2, so on until all points are paged over
    for (int row = firstRow; row < numRows; row += rowStride) {
        // Extact int4 from global
        int globalPoint = row + firstGlobalRow;
        int globalDim = firstCol + (firstGlobalCol / copiesPerStageRow);
        int globalPointIndex = (globalPoint * globalLeadingDim) + globalDim;

        // Store int4 to shared
        int swizzledAddress = SwizzleAddress(row, firstCol, copiesPerStageRow);

        if (Debug && blockIdx.x == 0) {
            // To check for coalesced accesses
            printf("globalPointIndex T%d Point %d: %d\n", threadIdx.x, row, globalPointIndex);
            printf("swizzledAddress T%d Point %d: %d\n", threadIdx.x, row, swizzledAddress);
        }
        // Don't write value in async version
        // sharedArray[swizzledAddress] = values;
        cuda::memcpy_async(sharedArray + swizzledAddress, globalArray + globalPointIndex,
                           (sizeof(int4)), pipe);
    }
}

__global__ void MatrixTransferKernel(int4* matrix, int kStride,
                                     unsigned long long* iterationCount) {
    unsigned int count = 0;

    // Pipeline init
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    extern __shared__ int4 sharedMem[];

    int4* BlockTile[pipelineDepth];

    for (int i = 0; i < pipelineDepth; ++i) {
        BlockTile[i] = sharedMem + (i * elementsPerStage / elementsPerInt4);
        if (threadIdx.x == 0 && Debug) {
            printf("Shared Memory addresses in stage %d are:\nTile: %p\n", i, BlockTile[i]);
        }
    }

    __syncthreads();

    // The first row this block needs to page in from the matrix
    int blockRow = blockIdx.y * rowsPerBlock;

    // Async pipeline direct global->shared
    // Fill pipeline
    int nextKToLoad = 0;
    for (int i = 0; i < pipelineDepth; ++i) {
        pipeline.producer_acquire();

        // Page in a stage worth
        LoadGlobalToSharedAsync(pipeline, matrix, BlockTile[i], blockRow, rowsPerBlock, nextKToLoad,
                                kStride);

        nextKToLoad += columnsPerStage;
        pipeline.producer_commit();
    }
    // Pipeline stage to consume next
    int pipelineIndex = 0;
    for (int kStart = 0; kStart < kStride; kStart += columnsPerStage) {
        cuda::pipeline_consumer_wait_prior<pipelineDepth - 1>(pipeline);

        // Thread scoped pipeline so must sync so we can read other thread's data
        __syncthreads();

        // Do work on items
        // Contrived but loop through all elements in shared and increment count

        __syncthreads();

        pipeline.consumer_release();

        // Queue up the next stage of the pipeline
        pipeline.producer_acquire();
        // Still queue up empty stage if all the data we need is in the pipeline so wait
        // doesn't block indefinitely at the end of the computation
        if (nextKToLoad < kStride) {
            // Page next stage in
            LoadGlobalToSharedAsync(pipeline, matrix, BlockTile[pipelineIndex], blockRow,
                                    rowsPerBlock, nextKToLoad, kStride);

            nextKToLoad += columnsPerStage;
        }
        pipeline.producer_commit();

        pipelineIndex = (pipelineIndex + 1) % pipelineDepth;
    }

    atomicAdd(iterationCount, count);
}
