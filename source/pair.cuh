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
        cudaMalloc(&d_pairs, sizeof(Pair) * maxPairs);
        cudaMalloc(&d_pairsFound, sizeof(unsigned long long));
        cudaMemset(d_pairsFound, 0, sizeof(unsigned long long));
        cudaMalloc(&d_pairsStored, sizeof(unsigned long long));
        cudaMemset(d_pairsStored, 0, sizeof(unsigned long long));
    }

    /** Release resources acquired by object. Must be called before object goes out of scope.
     *
     */
    __host__ void release() {
        if (d_pairs) {
            cudaFree(d_pairs);
            d_pairs = nullptr;
        }
        if (d_pairsFound) {
            cudaFree(d_pairsFound);
            d_pairsFound = nullptr;
        }
        if (d_pairsStored) {
            cudaFree(d_pairsStored);
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
        int old = atomicAdd(d_pairsFound, numPairs);
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
        transferPairCounts();
        std::vector<Pair> h_pairs(h_pairsStored);

        cudaMemcpy(h_pairs.data(), d_pairs, sizeof(Pair) * h_pairsStored, cudaMemcpyDeviceToHost);

        return h_pairs;
    }

    /** Sort the pairs in ascending order.
     *
     */
    __host__ void sort() {
        transferPairCounts();
        printf("Number of pairs found %llu\n", h_pairsFound);
        printf("Number of pairs stored %llu\n", h_pairsStored);
        thrust::sort(thrust::device, d_pairs, d_pairs + h_pairsStored);
        cudaGetLastError();
    }

    __host__ unsigned long long getPairsFound() { return h_pairsFound; }
    __host__ unsigned long long getPairsStored() { return h_pairsStored; }

   private:
    const unsigned long long maxPairs{};  // The max number of pairs that can be stored.
    unsigned long long
        h_pairsFound{};  // How many pairs were found (including when we ran out of memory).
    unsigned long long h_pairsStored{};   // How many pairs have been stored (on host).
    Pair* d_pairs{};                      // The actual pairs on the device.
    unsigned long long* d_pairsFound{};   // How many pairs have been found.
    unsigned long long* d_pairsStored{};  // How many pairs have been stored in memory.

    __host__ void transferPairCounts() {
        cudaMemcpy(&h_pairsFound, d_pairsFound, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairsStored, d_pairsStored, sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
    }
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
