/******************************************************************************
 * File:             ptxMma.h
 *
 * Author:           Brian Curless
 * Created:          08/23/24
 * Description:      Low level ptx instructions and constructs for performing MMA using tensor cores
 *****************************************************************************/

#ifndef PTXMMA_H_CSKBWHES
#define PTXMMA_H_CSKBWHES

#include <cuda.h>
#include <cuda_fp16.h>

#include <iostream>

namespace Mma {

inline constexpr bool Debug = false;

using InPrec = half;
using OutPrec = float;

struct mmaShape {
    int m{};
    int n{};
    int k{};
};

struct Coordinate {
    int row{};
    int col{};
};

// Declare a constant for reference elsewhere
// Dimensions of a fundamental mma operation in ptx
constexpr mmaShape dims{16, 8, 16};

// Represents one operand A, B, C, or D, of an mma operaetion that ldmatrix loads into registers.
// Each thread holds a piece of this operand. Input operands in half precision are packed in pairs
// into registers. Precision and fragment size dictate how many registers are required
template <typename T, int NumReg>
struct Fragment {
   public:
    T Registers[NumReg]{};

    __device__ void clear() {
        for (int j = 0; j < NumReg; j++) {
            Registers[j] = 0.0f;
        }
    }
};

// Making some aliases to make it easier to define specific functions taking only these template
// arguments due to the fact that they require inline ptx and aren't generic
using FragmentA_16x16 = Fragment<uint32_t, 4>;
using FragmentB_16x8 = Fragment<uint32_t, 2>;
using FragmentD_16x8 = Fragment<OutPrec, 4>;

__device__ uint32_t cvta_to_shared_u32(const void* pointer);
__device__ void loadAMatrix_16_16(const void* smem_row_start, FragmentA_16x16& A);
__device__ void loadBMatrix_16_8(const void* smem_row_start, FragmentB_16x8& B);
__device__ void mma_16_8_16(const FragmentA_16x16& A, const FragmentB_16x8& B,
                            const FragmentD_16x8& C, FragmentD_16x8& D);
__device__ Coordinate GetDElementCoordinate_16_8_float(Coordinate& baseCoord, int threadInWarp,
                                                       int dIndex);

/** Convert pointer to shared memory state space. This is converting our generic 64 bit address to
 * the shared memory state space. This means subtracting the base address of the shared space, and
 * then truncating to 32 bits. Since shared memory all fits into 32 bits this is safe. The generic
 * address space is 64 bits.
 *
 * \param pointer The generic pointer to convert
 *
 * \return 32-bit pointer to shared memory
 */
__device__ uint32_t cvta_to_shared_u32(const void* pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
}

/** Load row of shared memory into 16x16 Fragment.
 *
 *
 *
 * \param smem_row_start int4* to shared memory
 * \param A fragment to load into
 *
 */
__device__ void loadAMatrix_16_16(const void* smem_row_start, FragmentA_16x16& A) {
    //  Page into A
    //  Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(smem_row_start);

    if (Debug) {
        if (threadIdx.x == 0) {
            printf("Shared 32b Memory Address A 0x%x\n", smem_ptr);
        }
    }

    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=r"(A.Registers[0]), "=r"(A.Registers[1]), "=r"(A.Registers[2]), "=r"(A.Registers[3])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect A
        for (int i = 0; i < 4; i++) {
            half2* tempVal = reinterpret_cast<half2*>(A.Registers[i]);
            printf("Thread %d, %d: A%d=%f, A%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

/** Load row of shared memory into 16x8 Fragment.
 *
 *
 *
 * \param smem_row_start int4* to shared memory
 * \param B fragment to load into
 *
 */
__device__ void loadBMatrix_16_8(const void* smem_row_start, FragmentB_16x8& B) {
    //  Page into A
    //  Pointer to 128 bit row of data in shared memory
    uint32_t smem_ptr;

    smem_ptr = cvta_to_shared_u32(smem_row_start);

    if (Debug) {
        if (threadIdx.x == 0) {
            printf("Shared 32b Memory Address B 0x%x\n", smem_ptr);
        }
    }

    // To get this to work out like the example, you don't want to transpose here.
    // It seems there is an implied transpose for the B matrix when you execute mma
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
        "{ %0, %1 }, [%2];"
        : "=r"(B.Registers[0]), "=r"(B.Registers[1])
        : "r"(smem_ptr));

    if (Debug) {
        // Inspect B
        for (int i = 0; i < 2; i++) {
            half2* tempVal = reinterpret_cast<half2*>(&B.Registers[i]);
            printf("Thread %d, %d: B%d=%f, B%d=%f\n", threadIdx.x, threadIdx.y, i,
                   __half2float(tempVal->x), i, __half2float(tempVal->y));
        }
    }
}

/** Computes A*B+C=D
 *
 *
 *
 * \param A Input fragment operand
 * \param B Input fragment operand
 * \param C Input fragment operand
 * \param D Output fragment operand
 *
 */
__device__ void mma_16_8_16(const FragmentA_16x16& A, const FragmentB_16x8& B,
                            const FragmentD_16x8& C, FragmentD_16x8& D) {
    // 16x8x8 TC Operation
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7}, "
        " { %8, %9 }, "
        " { %10, %11, %12, %13 };"
        : "=f"(D.Registers[0]), "=f"(D.Registers[1]), "=f"(D.Registers[2]), "=f"(D.Registers[3])
        : "r"(A.Registers[0]), "r"(A.Registers[1]), "r"(A.Registers[2]), "r"(A.Registers[3]),
          "r"(B.Registers[0]), "r"(B.Registers[1]), "f"(C.Registers[0]), "f"(C.Registers[1]),
          "f"(C.Registers[2]), "f"(C.Registers[3]));

    if (Debug) {
        printf("Thread %d, %d: D0=%f, D1=%f, D2=%f, D3=%f\n", threadIdx.x, threadIdx.y,
               D.Registers[0], D.Registers[1], D.Registers[2], D.Registers[3]);
    }
}

/** Compute the global coordinates of a specific output register.
 *
 * \param baseCoord Upper left global coordinate of a single Mma operation's output matrix D
 * \param threadInWarp Warp lane (0..31)
 * \param dIndex Index of output element (0..4)
 *
 * \return Global coordinates of this element
 */
__device__ Coordinate GetDElementCoordinate_16_8_float(Coordinate& baseCoord, int threadInWarp,
                                                       int dIndex) {
    // TODO, this feels like I'm hard coding this...maybe it can be templated?
    int row, col;
    // First 8x8 matrix
    if (dIndex < 2) {
        row = baseCoord.row + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + dIndex;
    }
    // second 8x8 matrix
    else {
        row = baseCoord.row + 8 + (threadInWarp / 4);
        col = baseCoord.col + ((threadInWarp % 4) * 2) + (dIndex % 2);
    }
    return {row, col};
}
}  // namespace Mma
#endif /* end of include guard: PTXMMA_H_CSKBWHES */
