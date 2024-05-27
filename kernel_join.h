#ifndef KERNEL_JOIN_H
#define KERNEL_JOIN_H

#include <cuda_fp16.h>

#include "params.h"

__global__ void printMatrix(double* matrix, unsigned int nbElements);
__global__ void printMatrixTranspose(double* matrix, unsigned int size, unsigned int nbElements);
__global__ void printMatrixResult(double* matrix, unsigned int size, unsigned int nbElements);

__global__ void convertDataset(
    INPUT_DATA_TYPE* in,
    COMPUTE_TYPE* out,
    unsigned int nbPoints);

__global__ void preComputedSquaredCoordinates(
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* preComputeCoordinates,
    unsigned int nbQueryPoints);

__global__ void preComputedSquaredCoordinatesComplete(
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* preComputeCoordinates,
    unsigned int nbQueryPoints);

__global__ void transposeDataset(
    COMPUTE_TYPE* inputDataset,
    COMPUTE_TYPE* outputDataset,
    unsigned int nbQueryPoints);

__global__ void fillResultMatrix(
    ACCUM_TYPE* preComputedSquaredCoordinates,
    ACCUM_TYPE* resultMatrix,
    unsigned int nbQueryPoints);

__global__ void finishResultMatrix(
    ACCUM_TYPE* preComputedSquaredCoordinates,
    ACCUM_TYPE* resultMatrix,
    unsigned int nbQueryPoints,
    unsigned long long* cnt,
    ACCUM_TYPE* epsilon);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceCuda(
    unsigned int* nbQueryPoints,
    COMPUTE_TYPE* dataset,
    ACCUM_TYPE* epsilon,
    unsigned long long* cnt);


__global__ void distanceCalculationBruteForceCudaAlt(
        unsigned int* nbQueryPoints,
        COMPUTE_TYPE* dataset,
        ACCUM_TYPE* epsilon,
        unsigned long long* cnt);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceTensorBasic(
        unsigned int* nbQueryPoints,
        COMPUTE_TYPE* dataset,
        ACCUM_TYPE* epsilon,
        COMPUTE_TYPE* identityMatrix,
        unsigned long long* cnt);


__global__ void distanceCalculationBruteForceTensorHalfOpti(
        unsigned int* nbQueryPoints,
        COMPUTE_TYPE* dataset,
        ACCUM_TYPE* epsilon,
        unsigned long long* cnt,
        ACCUM_TYPE* preComputedSquaredCoordinates);

// Full summed version of mixed precision kernel
template <int BlockItemsY, int BlockItemsX, int BlockItemsK, int IterTileSize, int WmmaMd,
          int WmmaNd, int WmmaKd>
__device__ void distanceTCFullySummed(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                            ACCUM_TYPE* epsilon, unsigned long long* cnt,
                                            ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCFullySummed_16x16x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCFullySummed_32x8x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCFullySummed_8x32x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);

// Short circuitable version of mixed precision kernel
template <int Md, int Nd, int Kd>
__device__ void distanceTCShortCircuitable(unsigned int* nbQueryPoints, COMPUTE_TYPE* dataset,
                                            ACCUM_TYPE* epsilon, unsigned long long* cnt,
                                            ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCShortCircuitable_16x16x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCShortCircuitable_32x8x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);

__global__ void distanceTCShortCircuitable_8x32x16(unsigned int* nbQueryPoints,
                                                     COMPUTE_TYPE* dataset, ACCUM_TYPE* epsilon,
                                                     unsigned long long* cnt,
                                                     ACCUM_TYPE* preComputedSquaredCoordinates);
//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationBruteForceTensorDoubleOpti(
    unsigned int* nbQueryPoints,
    double* dataset,
    double* epsilon,
    unsigned long long* cnt,
    double* preComputedSquaredCoordinates);

#endif
