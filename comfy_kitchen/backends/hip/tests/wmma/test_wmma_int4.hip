// Save as test_wmma_int4.hip, compile and run separately
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

__global__ void test_int4_mma_layout()
{
        using Native4I = __attribute__((__vector_size__(16))) int;
        using Native8I = __attribute__((__vector_size__(32))) int;

        // A: each row gets its row number as value [0, 1, 2, ..., 15]
        // Pack 8 values per int32: each nibble = row number
        Native4I a_regs;
        int* a_ptr = (int*)&a_regs;
        for (int r = 0; r < 4; r++)
        {
                uint32_t packed = 0;
                for (int b = 0; b < 4; b++)
                {
                        int row0 = r * 4 + b * 2;  // two rows per byte
                        int row1 = r * 4 + b * 2 + 1;
                        packed |= ((row0 & 0xF) | ((row1 & 0xF) << 4)) << (b * 8);
                }
                a_ptr[r] = packed;
        }

        // B: first column = 1, rest = 0
        Native4I b_regs = {0x00000001, 0, 0, 0};

        Native8I c_regs = {0, 0, 0, 0, 0, 0, 0, 0};
        Native8I d_regs =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a_regs, 1, b_regs, c_regs, 1);

        int* d = (int*)&d_regs;

        if (threadIdx.x == 0)
        {
                printf("WMMA lane mapping test:\n");
                printf("A: rows 0-15 each = their row number\n");
                printf("B: only col 0 = 1, rest 0\n");
                printf("Expected: D[row, 0] = row_number (sum of 16, 1 per K)\n");
                for (int i = 0; i < 8; i++)
                {
                        printf("  d[%d] = %d\n", i, d[i]);
                }
        }
}

int main()
{
        hipSetDevice(0);
        test_int4_mma_layout<<<1, 32>>>();
        hipDeviceSynchronize();
        return 0;
}
