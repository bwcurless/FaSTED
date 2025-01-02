#ifndef ONDEMANDRASTERIZER_H
#define ONDEMANDRASTERIZER_H

#include <driver_types.h>

#include <cuda/semaphore>

#include "../matrix.cuh"
#include "../utils.cuh"

namespace Raster {

/** Rasterizes the layout of how work gets distributed to each thread block. It is important that
 * each block that is running at the same time operates on the same input data.
 * This increases the locality of data accesses, and dramatically increases the cache hits. This
 * routine organizes work into square chunks where each block is ideally looking at the same
 * candidate and query points at the same time. This class doesn't make any assumptions about the
 * execution order of thread blocks, it simply divies up the work as blocks request it.
 *
 */
class OnDemandRasterizer {
   private:
    unsigned long long numBlocksRows;  // How many rows of blocks there are
    unsigned long long numBlocksCols;  // How many columns of blocks there are
    unsigned int rasterSize;           // How big of a square to rasterize (width/height).
    unsigned long long rasterArea;     // The total number of blocks in a single raster.
    unsigned long long currentIndex;   // The next chunk to divy out.
    unsigned long long maxIndex;       // The maximum number of chunks that we should divy out.
    unsigned long long numRasterBlocksRows;  // How many rows of raster blocks there are.
    unsigned long long numRasterBlocksCols;  // How many columns of raster blocks there are.

   public:
    /** The initialize method must be called before using this class. The class must also be
     * manually cleaned up by calling release, as a copy of this object is passed into the cuda
     * kernel, and resources must not be released.
     */
    OnDemandRasterizer();

    virtual ~OnDemandRasterizer();

    /** Allocate any resources and prepare the GPU to reference this class. This needs to be invoked
     * by a kernel and synchronized before other kernel invocations can use it.
     *
     * \param numBlocksRows How many rows of blocks there are total in the computation.
     * \param numBlocksCols How many cols of blocks there are total in the computation.
     * \param rasterizeSize The height/width of the rasterized square.
     */
    __device__ void initialize(unsigned long long numBlocksRows, unsigned long long numBlocksCols,
                               unsigned int rasterSize) {
        this->numBlocksRows = numBlocksRows;
        this->numBlocksCols = numBlocksCols;
        this->rasterSize = rasterSize;
        rasterArea = rasterSize * rasterSize;
        // These should be dividing evenly, otherwise it all fails.
        numRasterBlocksRows = numBlocksRows / rasterSize;
        numRasterBlocksCols = numBlocksCols / rasterSize;

        maxIndex = numBlocksRows * numBlocksCols;
        currentIndex = 0;

        if (Debug) {
            printf("Raster Blocks Rows: %llu\n", numRasterBlocksRows);
            printf("Raster Blocks Cols: %llu\n", numRasterBlocksCols);
            printf("Raster Area: %llu\n", rasterArea);
            printf("Raster Size: %u\n", rasterSize);
        }
    }

    /** Gets the block coordinates for the next chunk of work that should be completed. This
     * effectively transforms a 1D layout of blocks into a 2D layout of blocks. As blocks execute,
     * they get their indeces. As block size changes, this method is unaffected. Each block knows
     * how large of a computation it is responsible for, this simply returns the next indexes.
     *
     * \returns The indices of the resulting computation that should be worked on next.
     */
    __device__ matrix::Coordinate nextChunk() {
        // Obtain the next index and increment it
        unsigned long long blockIndex =
            atomicAdd(&currentIndex, static_cast<unsigned long long>(1));
        // Can assume that the problem will be padded up to a multiple of the raster size.
        // This makes the math easy

        // Which n x n rasterized block to return.
        unsigned long long rasterBlockIndex = blockIndex / rasterArea;
        // Compute the base coordinates of the raster block
        unsigned int baseRow = (rasterBlockIndex / numRasterBlocksCols) * rasterSize;
        unsigned int baseCol = (rasterBlockIndex % numRasterBlocksCols) * rasterSize;

        // Index in the n x n rasterized block to return.
        unsigned long long localIndex = blockIndex % rasterArea;
        // Compute the local coordinates
        unsigned int localRow = localIndex / rasterSize;
        unsigned int localCol = localIndex % rasterSize;

        matrix::Coordinate coords(baseRow + localRow, baseCol + localCol);
        if (Debug) {
            printf("BaseRow is: %d LocalRow is: %d RasterIndex is: %llu Block Index is: %llu\n",
                   baseRow, localRow, rasterBlockIndex, blockIndex);
            printf("BaseCol is: %d LocalCol is: %d\n", baseCol, localCol);
            // printf("Chunk coordinates are: %d, %d\n", coords.row, coords.col);
        }
        return coords;
    }
};

}  // namespace Raster
#endif /* ONDEMANDRASTERIZER_H */
