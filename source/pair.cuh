/******************************************************************************
 * File:             pair.cuh
 *
 * Author:           Brian Curless
 * Created:          10/28/24
 * Description:      Represents the pairs of points that were found within a distance epsilon of
 *		     each other.
 *****************************************************************************/
#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/sort.h>

#include <iostream>

#include "utils.cuh"

namespace Pairs {

/** Two points that were found within a distance epsilon of each other.
 *
 * \param QueryPoint The index of the query point.
 * \param CandidatePoint The index of the candidate point.
 *
 */
struct Pair {
   public:
    int QueryPoint{};
    int CandidatePoint{};

    __device__ Pair() : QueryPoint{-1}, CandidatePoint{-1} {};

    __device__ Pair(int query, int candidate) : QueryPoint{query}, CandidatePoint{candidate} {};

    /** Compare in row-major order.
     *
     */
    __device__ bool operator<(const Pair& other) const {
        if (this->QueryPoint == other.QueryPoint) {
            return this->CandidatePoint < other.CandidatePoint;
        } else {
            return this->QueryPoint < other.QueryPoint;
        }
    }
};

/** Prints the pair to the output stream
 *
 */
__host__ std::ostream& operator<<(std::ostream& os, const Pair& obj) {
    os << obj.QueryPoint << ", " << obj.CandidatePoint << std::endl;
    return os;
}

/** A set containing all the pairs on the device. Multiple threads can safely request space to
 * store their pairs as they find them.
 *
 *
 */
class Pairs {
   public:
    Pairs(unsigned long long maxSize) : maxPairs{maxSize} {};

    /** Initialize object. Must call release before disposing of this object. Allocates
     * global memory on GPU, as well as memory on host.
     *
     */
    __host__ void init() {
        gpuErrchk(cudaMalloc(&d_pairs, sizeof(Pair) * maxPairs));
        gpuErrchk(cudaMalloc(&d_pairsFound, sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_pairsFound, 0, sizeof(unsigned long long)));
        gpuErrchk(cudaMalloc(&d_pairsStored, sizeof(unsigned long long)));
        gpuErrchk(cudaMemset(d_pairsStored, 0, sizeof(unsigned long long)));
    }

    /** Release resources acquired by object. Must be called before object goes out of scope.
     *
     */
    __host__ void release() {
        if (d_pairs) {
            gpuErrchk(cudaFree(d_pairs));
            d_pairs = nullptr;
        }
        if (d_pairsFound) {
            gpuErrchk(cudaFree(d_pairsFound));
            d_pairsFound = nullptr;
        }
        if (d_pairsStored) {
            gpuErrchk(cudaFree(d_pairsStored));
            d_pairsStored = nullptr;
        }
    }

    /** Request a place to store pairs to. Checks if there is space in the array. If there
     * is, then returns a pointer for where to store the first pair to.
     *
     * \param numPairs How many pairs you wish to store.
     *
     * \returns The pointer to store your pairs at. nullptr if failed to allocate enough space.
     */
    __device__ Pair* getSpace(unsigned int numPairs = 1) {
        unsigned long long old = atomicAdd(d_pairsFound, numPairs);
        // Failed to allocate enough space
        if (old + numPairs > maxPairs) {
            return nullptr;
        } else {
            // Keep track of how many pairs we actually stored.
            atomicAdd(d_pairsStored, numPairs);
            return d_pairs + old;
        }
    }

    /** Gets the array of pairs from the GPU. Transfer the pairs off the device and returns
     * them.
     *
     */
    __host__ std::vector<Pair> getPairs() {
        unsigned long long pairsStored = getPairsStored();
        std::vector<Pair> pairs(pairsStored);

        cudaMemcpy(pairs.data(), d_pairs, sizeof(Pair) * pairsStored, cudaMemcpyDeviceToHost);

        return pairs;
    }

    /** Sort the pairs in ascending order.
     *
     */
    __host__ void sort() {
        unsigned long long pairsFound = getPairsFound();
        unsigned long long pairsStored = getPairsStored();
        printf("Number of pairs found %llu\n", pairsFound);
        printf("Number of pairs stored %llu\n", pairsStored);
        thrust::sort(thrust::device, d_pairs, d_pairs + pairsStored);
        cudaGetLastError();
    }

    /** Get how many pairs the GPU found.
     *
     * \returns The number of pairs.
     */
    __host__ unsigned long long getPairsFound() {
        unsigned long long pairsFound;
        cudaMemcpy(&pairsFound, d_pairsFound, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        return pairsFound;
    }

    /** Get how many pairs the GPU was able to store to the buffer. Might have run out of space
     * while storing pairs. There is only so much GPU memory. This number might be less than the
     * number of pairs found.
     *
     * \returns The number of pairs stored.
     */
    __host__ unsigned long long getPairsStored() {
        unsigned long long pairsStored;
        cudaMemcpy(&pairsStored, d_pairsStored, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        return pairsStored;
    }

   private:
    const unsigned long long maxPairs{};  // The max number of pairs that can be stored.
    Pair* d_pairs{};                      // The actual pairs on the device.
    unsigned long long* d_pairsFound{};   // How many pairs have been found.
    unsigned long long* d_pairsStored{};  // How many pairs have been stored in memory.
};

/** Print all the pairs to the output stream. Copies all data off of device and puts them into the
 * output stream.
 *
 */
__host__ std::ostream& operator<<(std::ostream& os, Pairs& pairsObj) {
    auto pairs = pairsObj.getPairs();
    for (const Pair& pair : pairs) {
        os << pair;
    }
    return os;
}

}  // namespace Pairs
