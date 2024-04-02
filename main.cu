#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "dataset.h"
#include "gpu_join.h"
#include "main.h"
#include "omp.h"
#include "params.h"

int main(int argc, char* argv[]) {
    char filename[256];
    char* endptr;
    if (argc < 4) {
        fprintf(stderr,
                "Too few command line arguments. <pathToDataset> <epsilon> <searchMode> are "
                "required\n");
        return EXIT_FAILURE;
    }

    // Filepath is checked later when dataset is loaded
    strcpy(filename, argv[FILENAME_ARG]);

    // Validate that a valid double was passed in
    ACCUM_TYPE epsilon = strtod(argv[EPSILON_ARG], &endptr);
    std::cout << "[Main | Input] ~ Epsilon: " << epsilon << '\n';
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid argument, `%s' is not a valid epsilon\n", argv[EPSILON_ARG]);
        return EXIT_FAILURE;
    }

    // We should get an error when we go to use the search mode if it's invalid, no point in
    // checking if it's valid here
    unsigned int searchMode = atoi(argv[SEARCHMODE_ARG]);
    std::cout << "[Main | Input] ~ SearchMode: " << searchMode << '\n';

    unsigned int device = 0;
    if (argc == DEVICE_ARG + 1) {
        device = strtol(argv[DEVICE_ARG], &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid argument, `%s' is not a valid GPU device index\n",
                    argv[DEVICE_ARG]);
            return EXIT_FAILURE;
        }
    }

    std::cout << "[Main | Input] ~ Running on device: " << device << '\n';
    cudaSetDevice(device);

    /***** Import dataset *****/
    std::vector<std::vector<INPUT_DATA_TYPE> > inputVector;
    double tStartReadDataset = omp_get_wtime();
    importDataset(&inputVector, filename);
    double tEndReadDataset = omp_get_wtime();
    double timeReadDataset = tEndReadDataset - tStartReadDataset;
    std::cout << "[Main | Time] ~ Time to read the dataset: " << timeReadDataset << '\n';
    unsigned int nbQueryPoints = inputVector.size();

    INPUT_DATA_TYPE* database =
        new INPUT_DATA_TYPE[(nbQueryPoints + ADDITIONAL_POINTS) * COMPUTE_DIM];
    for (unsigned int i = 0; i < nbQueryPoints; ++i) {
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
            database[i * COMPUTE_DIM + j] = inputVector[i][j];
        }
        for (unsigned int j = INPUT_DATA_DIM; j < COMPUTE_DIM; ++j) {
            database[i * COMPUTE_DIM + j] = (INPUT_DATA_TYPE)0.0;
        }
    }
    for (unsigned int i = 0; i < ADDITIONAL_POINTS; ++i) {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j) {
            database[(nbQueryPoints + i) * COMPUTE_DIM + j] = (INPUT_DATA_TYPE)0.0;
        }
    }

    /***** Compute the distance similarity join *****/
    double tStartJoin = omp_get_wtime();
    uint64_t totalResult = 0;

    // TODO: Add your new search modes here
    switch (searchMode) {
        case SM_NVIDIA: {
            // We need to transpose the dataset for this searchMode
            INPUT_DATA_TYPE* datasetTranspose = new INPUT_DATA_TYPE[nbQueryPoints * COMPUTE_DIM];
            for (unsigned int i = 0; i < nbQueryPoints; ++i) {
                for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j) {
                    datasetTranspose[j * nbQueryPoints + i] = inputVector[i][j];
                }
                for (unsigned int j = INPUT_DATA_DIM; j < COMPUTE_DIM; ++j) {
                    datasetTranspose[j * nbQueryPoints + i] = (INPUT_DATA_TYPE)0.0;
                }
            }
            GPUJoinMainBruteForceNvidia(searchMode, device, database, datasetTranspose,
                                        &nbQueryPoints, &epsilon, &totalResult);
            break;
        }
        case SM_GPU:
        case SM_CUDA_ALT:
        case SM_TENSOR:
        case SM_TENSOR_OPTI:
        case SM_TENSOR_FS_16x16x16:
        case SM_TENSOR_FS_32x8x16:
        case SM_TENSOR_FS_8x32x16:
        case SM_TENSOR_SC_16x16x16:
        case SM_TENSOR_SC_32x8x16:
        case SM_TENSOR_SC_8x32x16: {
            GPUJoinMainBruteForce(searchMode, device, database, &nbQueryPoints, &epsilon,
                                  &totalResult);
            break;
        }
        case SM_CPU: {
            //            CPUJoinMainBruteForce(searchMode, database, &nbQueryPoints, &epsilon,
            //            &totalResult);
            break;
        }
        default: {
            std::cerr << "[Main] ~ Error: Unknown search mode: " << searchMode << "\n";
            return EXIT_FAILURE;
        }
    }

    double tEndJoin = omp_get_wtime();
    double timeJoin = tEndJoin - tStartJoin;
    std::cout << "[Main | Result] ~ Time to join: " << timeJoin << '\n';
    std::cout << "[Main | Result] ~ Total result set size: " << totalResult << '\n';

    std::ofstream outputResultFile;
    std::ifstream inputResultFile("tensor_brute-force.txt");
    outputResultFile.open("tensor_brute-force.txt", std::ios::out | std::ios::app);
    if (inputResultFile.peek() == std::ifstream::traits_type::eof()) {
        outputResultFile << "Dataset, epsilon, searchMode, executionTime, totalNeighbors, "
                            "inputDim, computeDim, "
                            "blockSize, warpPerBlock, computePrec, accumPrec\n";
    }
    outputResultFile << filename << ", " << epsilon << ", " << searchMode << ", " << timeJoin
                     << ", " << totalResult << ", " << INPUT_DATA_DIM << ", " << COMPUTE_DIM << ", "
                     << BLOCKSIZE << ", " << WARP_PER_BLOCK << ", " << COMPUTE_PREC << ", "
                     << ACCUM_PREC << std::endl;

    outputResultFile.close();
    inputResultFile.close();
}
