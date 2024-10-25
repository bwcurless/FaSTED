# cuSimSearch

## Overview
This is a FP16-FP32 mixed precision vector similarity search that is written for the A100 GPU. It operates in mixed precision with FP16 input data, and computes FP32 distances. Given two sets of vectors of N-dimensions and a distance epsilon $$\epsilon$$,
this routine computes the euclidian distance between every point. If two points are within $$\epsilon$$ of each other, their coordinates are output back to global memory and classified as a "Pair". The algorithm computes the distance between two vectors by performing a giant matrix multiplication of the candidate points times the query points. The summation of the squared dimensions of each point are added to the matrix product to compute the euclidian distance. 

## Optimizations
As the primary component of the algorithm is a large matrix multiplication, many of the optimizations were inspired by CUTLASS. On high dimensional datasets (> 2048D) with enough points to saturate the GPU, this algorithm is operating at 66% (200 TFLOPS) of peak utilization of the theoretical mixed-precision Tensor Core throughput (300 TFLOPS). Some of the optimizations made to achieve this performance are listed here:
- Inline PTX Tensor Core instructions for complete control over registers and shared memory.
- Warp Tiling.
  - Reuse of data paged from shared memory -> registers.
- Block Tiling.
  - Reuse of data paged from global memory -> shared memory by multiple warps.
- Coalesced global memory reads.
- Conflict free shared memory stores and loads from XOR swizzling of data in shared memory layout.
- Asynchronous copies from global direct to shared memory bypassing L1 Cache.
- Pipelining asynchronous copies to overlap computation with data movement.
- Aggressive loop unrolling by extensive use of compile time constants.
- L2 Cache tuned for 90% hit-rate by rasterizing thread block layout for higher locality.
- Efficient shared memory reductions.
- Occupancy tuning by adjusting shared memory and register usage.
- Vectorized instructions used to move data and compute with fewer instructions.
- Memory addresses are carefully aligned and padded throughout.

## Accuracy
Quantification of the accuracy of this mixed precision algorithm compared a double precision baseline is ongoing. Results will be posted here when they are acquired.
