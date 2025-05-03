# FaSTED

## Overview
This is a FP16-FP32 mixed precision vector similarity search that is written for the A100 GPU. It operates in mixed precision with FP16 input data, and computes FP32 distances. Given two sets of vectors of N-dimensions and a distance epsilon $$\epsilon$$,
this routine computes the euclidean distance between every point. If two points are within $$\epsilon$$ of each other, their coordinates are output back to global memory and classified as a "Pair". The algorithm computes the distance between two vectors by performing a giant matrix multiplication of the candidate points times the query points. The summation of the squared dimensions of each point are added to the matrix product to compute the euclidean distance. 

## ICPP2025 Submission
This algorithm was submitted to ICPP 2025 (International Conference on Parallel Processing). The experiments in the paper were run on the tag "128x128x64_Baseline". The changes made to the code in the "leave one out" TFLOPS study are all in the branches prefixed with "opt_".

## Optimizations
As the primary component of the algorithm is a large matrix multiplication, many of the optimizations were inspired by CUTLASS. On high dimensional datasets (> 2048D) with enough points to saturate the GPU, this algorithm is operating at 49% (154 TFLOPS) of peak utilization of the theoretical mixed-precision Tensor Core throughput (312 TFLOPS). Some of the optimizations made to achieve this performance are listed here:
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
- L2 Cache tuned for 90% hit-rate by thread block work queue for higher locality.
- Efficient shared memory reductions.
- Occupancy tuning by adjusting shared memory and register usage.
- Vectorized instructions used to move data and compute with fewer instructions.
- Memory addresses are carefully aligned and padded throughout.

## Experiments
The scripts used for the experiments in the paper are located in the "results/scripts" folder. 

### Accuracy
As this algorithm computes distances in FP16-32, the accuracy of the results, when compared with a purely FP64 baseline was quantified in the paper. Please refer to the paper for the exact results, however, the max accuracy loss across all datasets that we tested was 0.03%.

## How to Run
### Prerequisites
- You will need an Ampere generation GPU to run this code with a Compute Capability of 80. This code has only been tested on a PCIe A100, so your results may vary with different targets.
- CUDA 12.6.3 is required to compile and run this code.
### Compiling and Running
All experiments were run on NAU's Monsoon HPC Cluster. Some modifications will need to be made to in order to run on your own target. A brief overview is given here to assist others in making those changes:
1. There is a makefile in the source folder that can be used to build the "debug", "release", or "shared" targets. Many of the experiments were run with the "shared" library being invoked by python scripts.
2. An executable called "main" will be built in the "release" folder.
3. There are two ways to run the executable. The options for running it are as follow:
   - External Dataset (Any CSV style dataset with each row having a comma separated list of coordinates)
     - Directly pass the dataset you wish to run (i.e. './main "MyRealWorldDataset.txt" 0.001' to load a dataset in a file, and search it for neighbors with a radius of 0.001)  
   - Internally generated synthetic dataset (For ease of testing the software can generate synthetic datasets on the fly)
     - Pass the '-e' option to enable "exponential dataset" mode.
     - Pass the number of points, and the dimensionality of each point next. (i.e. './main -e 1000000 4096 0.001' for a synthetic dataset of 1M points, each 4096 dimensions, and a search radius of 0.001)

## Notes
It was found during the evaluation that the PCIe A100 GPU that was used in our development had its clock speeds being throttled due to excessive power consumption. This limited the throughput that we saw. Experiments are ongoing to investigate if reducing tile sizes, in order to increase occupancy and the number of eligible warps per scheduler, will improve throughput. The SXM version of the A100 is being looked at, and the results seem promising reducing the block tile size from 128x128x64 to 96x96x64.
