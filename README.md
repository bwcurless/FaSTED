# Brute-force Euclidean Distances

## How to use
- Open the `makefile` and make necessary changes to the existing `ampere`, `turing`, or `monsoon` rules according to where you wish to use this algorithm. Otherwise, add your own rule.
  - In particular, make sure that you have the NVIDIA Samples downloaded somewhere, as it is needed for the NVIDIA code that is included in this code. An other solution is to not use this specific code, i.e., to remove the files from the list of files to compile, and from the code (comment the code that call the functions, remove the includes, etc.).
- Open `params.h` and change the `#define` values according to the dataset/experiments you wish to run:
  - `INPUT_DATA_DIM` corresponds to the dimensionality of the dataset.
  - `COMPUTE_DIM` corresponds to the next multiple of 16 of `INPUT_DATA_DIM` in FP16, or the next multiple of 4 of `INPUT_DATA_DIM` in FP64.
  - `COMPUTE_PREC` corresponds to the precision of the computation, either FP16 or FP64 for tensor cores, or FP16, FP32, or FP64 for CUDA cores.
  - `ACCUM_PREC` corresponds to the precision of the accumulation. Must be greater than or equal to `COMPUTE_PREC`. When using tensor cores with FP16, you can only use FP16 or FP32 for accumulation.
- Compile the project using the appropriate rule (e.g, `make monsoon` to compile on Monsoon).
- Run the following command: `./main dataset_name epsilon algorithm [gpu_id]`
    - `dataset_name`: name of the dataset file (use a comma-separated kind of file, with one point per line).
    - `epsilon`: the distance threshold.
    - `algorithm`: the algorithm to use:
      - 1: CUDA cores
      - 2: CUDA cores, extended Euclidean distance formula
      - 3: Tensor cores (1 query point per warp at a time)
      - 4: Tensor cores (16 query points per warp at a time in FP16, 8 in FP64)
      - 5: NVIDIA algorithm (from GitHub)
      - 6: CPU algorithm *(WIP)*
    - `gpu_id`: Optional, ID of the GPU to use. If not specified, the first GPU will be used.