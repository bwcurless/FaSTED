#ifndef ONDEMANDRASTERIZER_H
#define ONDEMANDRASTERIZER_H

#include <driver_types.h>

#include <cuda/semaphore>
#include <cuda/std/optional>

#include "../matrix.cuh"
#include "../ptxMma.cuh"
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
    unsigned long long numChunksRows;  // How many rows of chunks there are
    unsigned long long numChunksCols;  // How many columns of chunks there are
    unsigned int rasterSize;           // How big of a square to rasterize (width/height).
    unsigned long long rasterArea;     // The total number of chunks in a single raster.
    unsigned long long currentIndex;   // The next chunk to divy out.
    unsigned long long maxChunks;      // The maximum number of chunks that we should divy out.
    unsigned long long numRasterChunksRows;  // How many rows of raster chunks there are.
    unsigned long long numRasterChunksCols;  // How many columns of raster chunks there are.
    Mma::mmaShape chunkShape;

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
     * \param numChunksRows How many rows of chunks there are total in the computation.
     * \param numChunksCols How many cols of chunks there are total in the computation.
     * \param rasterizeSize The height/width of the rasterized square.
     * \param chunkShape The height/width of a single chunk.
     */
    __device__ void initialize(unsigned long long numChunksRows, unsigned long long numChunksCols,
                               unsigned int rasterSize, Mma::mmaShape chunkShape) {
        this->numChunksRows = numChunksRows;
        this->numChunksCols = numChunksCols;
        this->rasterSize = rasterSize;
        this->chunkShape = chunkShape;

        rasterArea = rasterSize * rasterSize;
        // These should be dividing evenly, otherwise it all fails.
        numRasterChunksRows = numChunksRows / rasterSize;
        numRasterChunksCols = numChunksCols / rasterSize;

        maxChunks = numChunksRows * numChunksCols;
        currentIndex = 0;

        if (Debug) {
            printf("Raster Chunks Rows: %llu\n", numRasterChunksRows);
            printf("Raster Chunks Cols: %llu\n", numRasterChunksCols);
            printf("Raster Area: %llu\n", rasterArea);
            printf("Raster Size: %u\n", rasterSize);
        }
    }

    /** Gets the coordinates for the next chunk of work that should be completed. As blocks execute,
     * they get chunks of coordinates to work on from this method.
     *
     * \returns The coordinates of the chunk that should be worked on next.
     */
    __device__ cuda::std::optional<matrix::Coordinate> nextChunk() {
        // Obtain the next index and increment it
        unsigned long long chunkIndex = atomicAdd(&currentIndex, 1);
        if (chunkIndex >= maxChunks) {
            return cuda::std::nullopt;
        }
        // Can assume that the problem will be padded up to a multiple of the raster size.
        // This makes the math easy

        // Which n x n rasterized chunk to return.
        unsigned long long rasterChunkIndex = chunkIndex / rasterArea;
        // Compute the base coordinates of the raster chunk
        unsigned int baseRow = (rasterChunkIndex / numRasterChunksCols) * rasterSize;
        unsigned int baseCol = (rasterChunkIndex % numRasterChunksCols) * rasterSize;

        // Index in the n x n rasterized chunk to return.
        unsigned long long localIndex = chunkIndex % rasterArea;
        // Compute the local coordinates
        unsigned int localRow = localIndex / rasterSize;
        unsigned int localCol = localIndex % rasterSize;

        matrix::Coordinate coords((baseRow + localRow) * chunkShape.m,
                                  (baseCol + localCol) * chunkShape.n);
        if (Debug) {
            printf("BaseRow is: %d LocalRow is: %d RasterIndex is: %llu Chunk Index is: %llu\n",
                   baseRow, localRow, rasterChunkIndex, chunkIndex);
            printf("BaseCol is: %d LocalCol is: %d\n", baseCol, localCol);
            // printf("Chunk coordinates are: %d, %d\n", coords.row, coords.col);
        }
        return coords;
    }
};

}  // namespace Raster
#endif /* ONDEMANDRASTERIZER_H */
