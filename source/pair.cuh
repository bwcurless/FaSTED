/******************************************************************************
 * File:             pair.cuh
 *
 * Author:           Brian Curless
 * Created:          10/28/24
 * Description:      Represents the pairs of points that were found within a distance epsilon of
 *		     each other.
 *****************************************************************************/
#pragma once

#include <driver_types.h>

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

    Pair() : QueryPoint{-1}, CandidatePoint{-1} {};

    Pair(int query, int candidate) : QueryPoint{query}, CandidatePoint{candidate} {};

    void print() const { printf("%d, %d\n", QueryPoint, CandidatePoint); }
};

/** A set containing all the pairs on the device. Multiple threads can safely request space to store
 * their pairs as they find them.
 *
 *
 */
class Pairs {
   public:
    Pairs(int maxSize) : maxSize{maxSize} {};

    /** Initialize object. Must call release before disposing of this object. Allocates
     * global memory on GPU, as well as memory on host.
     *
     */
    __host__ void init() {
        cudaMalloc(&d_pairs, sizeof(Pair) * maxSize);
        cudaMalloc(&d_numPairs, sizeof(int));
        cudaMemset(d_numPairs, 0, 1);
    }

    /** Release resources acquired by object. Must be called before object goes out of scope.
     *
     */
    __host__ void release() {
        if (d_pairs) {
            cudaFree(d_pairs);
            d_pairs = nullptr;
        }
        if (d_numPairs) {
            cudaFree(d_numPairs);
            d_numPairs = nullptr;
        }
    }

    /** Request a place to store pairs to. Checks if there is space in the array. If there
     * is, then returns a pointer for where to store the first pair to.
     *
     * \param numPairs How many pairs you wish to store.
     *
     * \returns The pointer to store your pairs at. nullptr if failed to allocate enough space.
     */
    __device__ Pair* getSpace(int numPairs = 1) {
        int old = atomicAdd(d_numPairs, numPairs);
        // Failed to allocate enough space
        if (old + numPairs > maxSize) {
            printf("Ran out of room!\n");
            return nullptr;
        } else {
            return d_pairs + old;
        }
    }

    /** Gets the array of pairs from the GPU. Transfer the pairs off the device and returns them.
     *
     */
    __host__ std::vector<Pair> getPairs() {
        int h_numPairs;
        cudaMemcpy(&h_numPairs, d_numPairs, sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<Pair> h_pairs(h_numPairs);

        cudaMemcpy(h_pairs.data(), d_pairs, sizeof(Pair) * h_numPairs, cudaMemcpyDeviceToHost);

        return h_pairs;
    }

    /** Print the pairs out.
     *
     */
    __host__ void print() {
        auto pairs = getPairs();
        for (const Pair& pair : pairs) {
            pair.print();
        }
        printf("Pairs Max Size: %d, Current Size: %d\n", maxSize, pairs.size());
    }

   private:
    const int maxSize{};  // The max number of pairs that can be stored.
    Pair* d_pairs{};      // The actual pairs on the device.
    int* d_numPairs{};    // How many pairs have been stored.
};
}  // namespace Pairs
