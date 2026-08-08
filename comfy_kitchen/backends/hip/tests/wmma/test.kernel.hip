#include <stdio.h>

#include <rocwmma/rocwmma.hpp>

__global__ void test_wmma_int32_output_layout()
{
        if (threadIdx.x >= 32) return;

        using Native4I = __attribute__((__vector_size__(16))) int;
        using Native8I = __attribute__((__vector_size__(32))) int;

        Native4I a, b;
        Native8I c = {0, 0, 0, 0, 0, 0, 0, 0};

        int tid = threadIdx.x;

        // Each thread puts 1 in a unique byte position across all registers
        for (int reg = 0; reg < 4; reg++)
        {
                ((int*)&a)[reg] = 0;
        }
        // Thread 0: byte 0 of reg 0, Thread 1: byte 1 of reg 0, etc.
        int my_byte = tid;
        if (my_byte < 16)
        {  // First 16 threads cover A[0] and A[1]
                int reg = my_byte / 4;
                int byte_in_reg = my_byte % 4;
                ((unsigned char*)&a)[reg * 4 + byte_in_reg] = 1;
        }

        for (int reg = 0; reg < 4; reg++)
        {
                ((int*)&b)[reg] = 0x01010101;
        }

        Native8I d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a, 1, b, c, 1);

        printf("T%02d: [", tid);
        for (int i = 0; i < 8; i++)
        {
                printf("%d ", ((int*)&d)[i]);
        }
        printf("]\n");
}

__global__ void test_wmma_byte_mapping()
{
        if (threadIdx.x != 0) return;  // Just thread 0

        using Native4I = __attribute__((__vector_size__(16))) int;
        using Native8I = __attribute__((__vector_size__(32))) int;

        // Test each byte position in A[0] individually
        for (int test_byte = 0; test_byte < 4; test_byte++)
        {
                Native4I a, b;
                Native8I c = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int i = 0; i < 4; i++)
                        ((int*)&a)[i] = 0;
                ((unsigned char*)&a)[test_byte] = 1;  // Set just one byte

                for (int i = 0; i < 4; i++)
                        ((int*)&b)[i] = 0x01010101;

                Native8I d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a, 1, b, c, 1);

                printf("A byte %d set -> D: [", test_byte);
                for (int i = 0; i < 8; i++)
                {
                        printf("%08x ", ((int*)&d)[i]);
                }
                printf("]\n");
        }

        // Also test which accumulator registers get which A registers
        for (int test_reg = 0; test_reg < 4; test_reg++)
        {
                Native4I a, b;
                Native8I c = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int i = 0; i < 4; i++)
                        ((int*)&a)[i] = 0;
                ((int*)&a)[test_reg] = 0x01010101;  // All bytes in this register = 1

                for (int i = 0; i < 4; i++)
                        ((int*)&b)[i] = 0x01010101;

                Native8I d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a, 1, b, c, 1);

                printf("A reg %d all 1s -> D: [", test_reg);
                for (int i = 0; i < 8; i++)
                {
                        printf("%08x ", ((int*)&d)[i]);
                }
                printf("]\n");
        }
}

__global__ void test_wmma_identity()
{
        if (threadIdx.x >= 32) return;

        using Native4I = __attribute__((__vector_size__(16))) int;
        using Native8I = __attribute__((__vector_size__(32))) int;

        Native4I a, b;
        Native8I c = {0, 0, 0, 0, 0, 0, 0, 0};

        int tid = threadIdx.x;

        // Simple test: put 1 in first byte position only
        for (int reg = 0; reg < 4; reg++)
        {
                ((int*)&a)[reg] = (tid == 0 && reg == 0) ? 0x00000001 : 0;
        }
        for (int reg = 0; reg < 4; reg++)
        {
                ((int*)&b)[reg] = 0x01010101;  // all ones
        }

        Native8I d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a, 1, b, c, 1);

        printf("T%02d: D=[%08x %08x %08x %08x %08x %08x %08x %08x]\n", tid, ((int*)&d)[0],
               ((int*)&d)[1], ((int*)&d)[2], ((int*)&d)[3], ((int*)&d)[4], ((int*)&d)[5],
               ((int*)&d)[6], ((int*)&d)[7]);
}

__global__ void test_wmma_mapping_detailed()
{
        if (threadIdx.x >= 8) return;  // Just first 8 threads for clarity

        using Native4I = __attribute__((__vector_size__(16))) int;
        using Native8I = __attribute__((__vector_size__(32))) int;

        Native4I a, b;
        Native8I c = {0, 0, 0, 0, 0, 0, 0, 0};

        int tid = threadIdx.x;

        // Put unique values in each byte position to trace mapping
        // Each register gets a different base value
        // Each byte within a register gets tid*4 + byte_offset
        for (int reg = 0; reg < 4; reg++)
        {
                int32_t val = 0;
                for (int byte_pos = 0; byte_pos < 4; byte_pos++)
                {
                        int unique_val =
                            tid * 4 + reg * 16 + byte_pos;  // Globally unique per thread
                        val |= (unique_val & 0xFF) << (byte_pos * 8);
                }
                ((int*)&a)[reg] = val;
        }

        // B is identity: each register has 0x01010101 in its bytes
        for (int i = 0; i < 4; i++)
        {
                ((int*)&b)[i] = 0x01010101;
        }

        Native8I d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(1, a, 1, b, c, 1);

        printf("T%02d A: [%08x %08x %08x %08x]\n", tid, ((int*)&a)[0], ((int*)&a)[1], ((int*)&a)[2],
               ((int*)&a)[3]);
        printf("T%02d D: [%08x %08x %08x %08x %08x %08x %08x %08x]\n", tid, ((int*)&d)[0],
               ((int*)&d)[1], ((int*)&d)[2], ((int*)&d)[3], ((int*)&d)[4], ((int*)&d)[5],
               ((int*)&d)[6], ((int*)&d)[7]);
}
int main()
{
        hipSetDevice(0);

        test_wmma_mapping_detailed<<<1, 1>>>();
        test_wmma_identity<<<1, 1>>>();
        test_wmma_byte_mapping<<<1, 1>>>();
        test_wmma_int32_output_layout<<<1, 1>>>();
        printf("All tests launched. Fill in the WMMA intrinsics.\n");
        return 0;
}
