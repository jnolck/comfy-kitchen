// test_svdquant_mm.cpp
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <rocwmma/rocwmma.hpp>
#include <vector>

constexpr int kGroupSize = 64;

constexpr int kStages = 3;
constexpr int kWarpM = 16;
constexpr int kNUnroll = 2;
constexpr int kWarpN = kNUnroll * 16;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kNumWarps = 8;
constexpr int kBlockM = 32;
constexpr int kBlockN = 128;
constexpr int kBlockKInt8 = kGroupSize;
constexpr int kBlockKBytes = kGroupSize / 2;
constexpr int kTileInterleave = 4;
constexpr int kThreadsPerBlock = kNumWarps * 32;

template <typename T>
__device__ __forceinline__ float to_fp32(T v);
template <>
__device__ __forceinline__ float to_fp32<__hip_bfloat16>(__hip_bfloat16 v)
{
        return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_fp32<__half>(__half v)
{
        return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_fp32(float v);
template <>
__device__ __forceinline__ __hip_bfloat16 from_fp32<__hip_bfloat16>(float v)
{
        return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ __half from_fp32<__half>(float v)
{
        return __float2half(v);
}

__device__ int8_t unpack_int4_to_int8(int8_t packed, int idx)
{
        int val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        if (val >= 8) val -= 16;
        return static_cast<int8_t>(val);
}

// Copy your svdquant_scaled_mm_w4a4_kernel here
template <typename OutType, bool kActUnsigned, bool kTilePacked, bool kSharedScale, bool kFuseLora>
__global__ void svdquant_scaled_mm_w4a4_kernel(
    const int8_t* __restrict__ act, const int8_t* __restrict__ wgt,
    const OutType* __restrict__ ascales, const OutType* __restrict__ wscales,
    const OutType* __restrict__ lora_act_in, const OutType* __restrict__ lora_up,
    const OutType* __restrict__ bias, OutType* __restrict__ out, int M, int N, int K, int R)
{
        if constexpr (!kFuseLora)
        {
                (void)lora_act_in;
                (void)lora_up;
                (void)R;
        }
        (void)kActUnsigned;

        const int cta_m = blockIdx.y * kBlockM;
        const int cta_n = blockIdx.x * kBlockN;
        const int cta_n_tile = blockIdx.x;
        const int warp_id = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int warp_m = warp_id & (kWarpsM - 1);
        const int warp_n = warp_id / kWarpsM;
        const int warp_m_base = cta_m + warp_m * kWarpM;
        const int warp_n_base = cta_n + warp_n * kWarpN;

        // AMD-native thread mapping for 16×16 WMMA
        const int tid_in_col = lane & 7;  // 0..7  (column within 16-col tile)
        const int groupID = lane >> 3;    // 0..3  (row group within warp)

        const int K_half = K / 2;
        const int num_groups = K / kGroupSize;

        // Each thread accumulates 4 rows × 4 columns
        float acc_fp32[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                        acc_fp32[r][c] = 0.f;

        // Shared memory
        __shared__ alignas(16) int8_t smem_A[kStages][kBlockM * kBlockKInt8];
        __shared__ alignas(16) int8_t smem_B[kStages][kBlockN * kBlockKInt8];
        __shared__ alignas(16) int32_t smem_acc[kBlockM * kBlockN];
        __shared__ alignas(16) OutType smem_WS[kSharedScale ? kBlockN : 1];

        // ---- Load B tile (unpack int4→int8) ----
        auto load_B = [&](int g, int stage)
        {
                if (g >= num_groups) return;
                const int base_byte = g * kBlockKBytes;
                for (int t = threadIdx.x; t < kBlockN * kBlockKInt8; t += kThreadsPerBlock)
                {
                        int n_row = t / kBlockKInt8;
                        int k_col = t % kBlockKInt8;
                        int n_global = cta_n + n_row;
                        int8_t val = 0;
                        if (n_global < N)
                        {
                                int byte_idx = base_byte + k_col / 2;
                                int nibble = k_col & 1;
                                if constexpr (kTilePacked)
                                {
                                        int n_quad = n_row >> 2;
                                        int n_lane = n_row & 3;
                                        const int8_t* src =
                                            wgt +
                                            (cta_n_tile * num_groups + g) *
                                                (kBlockN * kBlockKBytes) +
                                            n_quad * (kTileInterleave * kBlockKBytes) +
                                            n_lane * kBlockKBytes + byte_idx;
                                        val = unpack_int4_to_int8(*src, nibble);
                                }
                                else
                                {
                                        val = unpack_int4_to_int8(wgt[n_global * K_half + byte_idx],
                                                                  nibble);
                                }
                        }
                        smem_B[stage][t] = val;
                }
        };

        // ---- Load A tile (unpack int4→int8) ----
        auto load_A = [&](int g, int stage)
        {
                if (g >= num_groups) return;
                const int base_byte = g * kBlockKBytes;
                for (int t = threadIdx.x; t < kBlockM * kBlockKInt8; t += kThreadsPerBlock)
                {
                        int m_row = t / kBlockKInt8;
                        int k_col = t % kBlockKInt8;
                        int m_global = cta_m + m_row;
                        int8_t val = 0;
                        if (m_global < M)
                        {
                                int byte_idx = base_byte + k_col / 2;
                                int nibble = k_col & 1;
                                val =
                                    unpack_int4_to_int8(act[m_global * K_half + byte_idx], nibble);
                        }
                        smem_A[stage][t] = val;
                }
        };

        // ---- Load weight scales into smem ----
        auto load_WS = [&](int g)
        {
                if constexpr (!kSharedScale)
                {
                        (void)g;
                        return;
                }
                if (g >= num_groups) return;
                for (int t = threadIdx.x; t < kBlockN; t += kThreadsPerBlock)
                {
                        int n_global = cta_n + t;
                        if (n_global < N)
                        {
                                if constexpr (kTilePacked)
                                        smem_WS[t] =
                                            wscales[(cta_n_tile * num_groups + g) * kBlockN + t];
                                else
                                        smem_WS[t] = wscales[g * N + n_global];
                        }
                        else
                        {
                                smem_WS[t] = OutType{};
                        }
                }
        };

// Prime pipeline
#pragma unroll
        for (int s = 0; s < kStages - 1; s++)
        {
                load_A(s, s);
                load_B(s, s);
        }
        load_WS(0);
        __syncthreads();

        // ---- Main loop over K groups ----
        for (int g = 0; g < num_groups; g++)
        {
                int next_g = g + kStages - 1;
                if (next_g < num_groups)
                {
                        int ns = (g + kStages - 1) % kStages;
                        load_A(next_g, ns);
                        load_B(next_g, ns);
                }
                load_WS(g + kStages - 1);
                __syncthreads();

                int stage = g % kStages;

                // ---- WMMA: int8 16×16×16, 4 K-slices, 2 N-tiles ----
                using A_frag =
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t, rocwmma::row_major>;
                using B_frag =
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t, rocwmma::col_major>;
                using AccI32 = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t>;

                AccI32 acc_i32[kNUnroll];
#pragma unroll
                for (int c = 0; c < kNUnroll; c++)
#pragma unroll
                        for (int i = 0; i < 8; i++)
                                acc_i32[c].x[i] = 0;

#pragma unroll
                for (int ks = 0; ks < 4; ks++)
                {
                        A_frag a_frag;
                        rocwmma::load_matrix_sync(
                            a_frag, &smem_A[stage][warp_m * kWarpM * kBlockKInt8 + ks * 16],
                            kBlockKInt8);

#pragma unroll
                        for (int c = 0; c < kNUnroll; c++)
                        {
                                B_frag b_frag;
                                rocwmma::load_matrix_sync(
                                    b_frag,
                                    &smem_B[stage]
                                           [(warp_n * kWarpN + c * 16) * kBlockKInt8 + ks * 16],
                                    kBlockKInt8);
                                rocwmma::mma_sync(acc_i32[c], a_frag, b_frag, acc_i32[c]);
                        }
                }

// Store int32 accumulators to smem (row-major, 16×32 per warp)
#pragma unroll
                for (int c = 0; c < kNUnroll; c++)
                {
                        rocwmma::store_matrix_sync(
                            &smem_acc[warp_m * kWarpM * kBlockN + warp_n * kWarpN + c * 16],
                            acc_i32[c], kBlockN, rocwmma::mem_row_major);
                }
                __syncthreads();

// ---- Dequant: multiply by per-group scales ----
#pragma unroll
                for (int r = 0; r < 4; r++)
                {
                        int row_local = warp_m * kWarpM + groupID * 4 + r;
                        int row_global = cta_m + row_local;
                        float as =
                            (row_global < M) ? to_fp32<OutType>(ascales[g * M + row_global]) : 0.f;

#pragma unroll
                        for (int c = 0; c < kNUnroll; c++)
                        {
                                int col_local = warp_n * kWarpN + c * 16 + tid_in_col * 2;
                                int col0_global = cta_n + col_local;
                                int col1_global = col0_global + 1;

                                int32_t v0 = smem_acc[row_local * kBlockN + col_local];
                                int32_t v1 = smem_acc[row_local * kBlockN + col_local + 1];

                                float ws0, ws1;
                                if constexpr (kSharedScale)
                                {
                                        ws0 = (col0_global < N)
                                                  ? to_fp32<OutType>(smem_WS[col_local])
                                                  : 0.f;
                                        ws1 = (col1_global < N)
                                                  ? to_fp32<OutType>(smem_WS[col_local + 1])
                                                  : 0.f;
                                }
                                else
                                {
                                        ws0 = (col0_global < N)
                                                  ? to_fp32<OutType>(wscales[g * N + col0_global])
                                                  : 0.f;
                                        ws1 = (col1_global < N)
                                                  ? to_fp32<OutType>(wscales[g * N + col1_global])
                                                  : 0.f;
                                }

                                acc_fp32[r][c * 2 + 0] += (float)v0 * as * ws0;
                                acc_fp32[r][c * 2 + 1] += (float)v1 * as * ws1;
                        }
                }
                __syncthreads();
        }

// ---- Write output ----
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
                int row_global = warp_m_base + groupID * 4 + r;
                if (row_global >= M) continue;

#pragma unroll
                for (int c = 0; c < kNUnroll; c++)
                {
                        int col0 = warp_n_base + c * 16 + tid_in_col * 2;
                        int col1 = col0 + 1;

                        float b0 = (bias && col0 < N) ? to_fp32<OutType>(bias[col0]) : 0.f;
                        float b1 = (bias && col1 < N) ? to_fp32<OutType>(bias[col1]) : 0.f;

                        if (col0 < N)
                                out[row_global * N + col0] =
                                    from_fp32<OutType>(acc_fp32[r][c * 2 + 0] + b0);
                        if (col1 < N)
                                out[row_global * N + col1] =
                                    from_fp32<OutType>(acc_fp32[r][c * 2 + 1] + b1);
                }
        }
}
// CPU reference for SVDQuant
void svdquant_mm_cpu(const int8_t* act,     // (M, K/2) packed int4
                     const int8_t* wgt,     // (N, K/2) packed int4
                     const float* ascales,  // (K/G, M)
                     const float* wscales,  // (K/G, N)
                     float* out,            // (M, N)
                     int M, int N, int K, int G)
{
        int K_half = K / 2;
        int num_groups = K / G;

        for (int m = 0; m < M; m++)
        {
                for (int n = 0; n < N; n++)
                {
                        float acc = 0.0f;
                        for (int g = 0; g < num_groups; g++)
                        {
                                float as = ascales[g * M + m];
                                float ws = wscales[g * N + n];
                                for (int k = 0; k < G; k++)
                                {
                                        int byte_idx = g * (G / 2) + k / 2;
                                        int nibble = k & 1;

                                        int a_val =
                                            (act[m * K_half + byte_idx] >> (nibble * 4)) & 0xF;
                                        if (a_val >= 8) a_val -= 16;

                                        int w_val =
                                            (wgt[n * K_half + byte_idx] >> (nibble * 4)) & 0xF;
                                        if (w_val >= 8) w_val -= 16;

                                        acc += (float)(a_val * w_val) * as * ws;
                                }
                        }
                        out[m * N + n] = acc;
                }
        }
}

int main()
{
        constexpr int M = 32, N = 128, K = 64, G = 64;  // Single group

        printf("=== SVDQuant scaled_mm Test ===\n");
        printf("M=%d N=%d K=%d G=%d\n\n", M, N, K, G);

        std::vector<int8_t> h_act(M * K / 2);
        std::vector<int8_t> h_wgt(N * K / 2);
        std::vector<float> h_ascales(M * K / G);
        std::vector<float> h_wscales(N * K / G);
        std::vector<__half> h_out_gpu(M * N);
        std::vector<float> h_out_cpu(M * N);

        // Test 1: All ones - simple verification
        printf("Test 1: All ones\n");
        // act: all 0x11 (both nibbles = 1)
        // wgt: all 0x11 (both nibbles = 1)
        // scales: all 1.0
        // Expected: each int4=1, 64 pairs of 1*1=1, scale 1*1=1
        // Total per output = 64 * 1 * 1 * 1 = 64
        memset(h_act.data(), 0x11, M * K / 2);
        memset(h_wgt.data(), 0x11, N * K / 2);
        std::fill(h_ascales.begin(), h_ascales.end(), 1.0f);
        std::fill(h_wscales.begin(), h_wscales.end(), 1.0f);

        // CPU
        svdquant_mm_cpu(h_act.data(), h_wgt.data(), h_ascales.data(), h_wscales.data(),
                        h_out_cpu.data(), M, N, K, G);
        printf("  CPU out[0,0] = %.1f (expected 64.0)\n", h_out_cpu[0]);

        // GPU
        int8_t *d_act, *d_wgt;
        __half *d_ascales, *d_wscales, *d_out;
        hipMalloc(&d_act, M * K / 2);
        hipMalloc(&d_wgt, N * K / 2);
        hipMalloc(&d_ascales, M * K / G * sizeof(__half));
        hipMalloc(&d_wscales, N * K / G * sizeof(__half));
        hipMalloc(&d_out, M * N * sizeof(__half));

        hipMemcpy(d_act, h_act.data(), M * K / 2, hipMemcpyHostToDevice);
        hipMemcpy(d_wgt, h_wgt.data(), N * K / 2, hipMemcpyHostToDevice);

        // Convert float scales to __half
        std::vector<__half> h_ascales_half(M * K / G);
        std::vector<__half> h_wscales_half(N * K / G);
        for (int i = 0; i < M * K / G; i++)
                h_ascales_half[i] = __float2half(h_ascales[i]);
        for (int i = 0; i < N * K / G; i++)
                h_wscales_half[i] = __float2half(h_wscales[i]);

        hipMemcpy(d_ascales, h_ascales_half.data(), M * K / G * sizeof(__half),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_wscales, h_wscales_half.data(), N * K / G * sizeof(__half),
                  hipMemcpyHostToDevice);

        // Launch kernel (kTilePacked=false, kSharedScale=false, kActUnsigned=false)
        dim3 grid(1, 1);
        dim3 block(256);
        svdquant_scaled_mm_w4a4_kernel<__half, false, false, false, false><<<grid, block, 0, 0>>>(
            d_act, d_wgt, d_ascales, d_wscales, nullptr, nullptr, nullptr, d_out, M, N, K, 0);

        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess)
        {
                printf("  GPU ERROR: %s\n", hipGetErrorString(err));
                return 1;
        }

        hipMemcpy(h_out_gpu.data(), d_out, M * N * sizeof(__half), hipMemcpyHostToDevice);

        // Check results
        int errors = 0;
        float max_diff = 0;
        for (int i = 0; i < M * N && errors < 10; i++)
        {
                float gpu = __half2float(h_out_gpu[i]);
                float cpu = h_out_cpu[i];
                float diff = fabsf(gpu - cpu);
                if (diff > 1.0f)
                {
                        printf("  ERROR [%d,%d]: gpu=%.1f cpu=%.1f diff=%.1f\n", i / N, i % N, gpu,
                               cpu, diff);
                        errors++;
                }
                if (diff > max_diff) max_diff = diff;
        }

        if (errors == 0)
        {
                printf("  PASS (max diff = %.2f)\n", max_diff);
        }
        else
        {
                printf("  FAIL: %d errors (max diff = %.2f)\n", errors, max_diff);
        }

        // Test 2: Varying values
        printf("\nTest 2: Varying values\n");
        // act: row r, col k = r + k (as int4, clamped to [-7,7])
        // wgt: all 1s
        // scales: ascale = 2.0, wscale = 0.5
        // Expected: sum over k of act_val * 1 * 2.0 * 0.5 = sum(act_val)

        for (int m = 0; m < M; m++)
        {
                for (int k = 0; k < K / 2; k++)
                {
                        int v0 = (m + k * 2) % 15;
                        if (v0 > 7) v0 -= 16;  // sign extend to [-8,7]
                        int v1 = (m + k * 2 + 1) % 15;
                        if (v1 > 7) v1 -= 16;
                        h_act[m * K / 2 + k] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
                }
        }
        memset(h_wgt.data(), 0x11, N * K / 2);  // all 1s
        std::fill(h_ascales.begin(), h_ascales.end(), 2.0f);
        std::fill(h_wscales.begin(), h_wscales.end(), 0.5f);

        svdquant_mm_cpu(h_act.data(), h_wgt.data(), h_ascales.data(), h_wscales.data(),
                        h_out_cpu.data(), M, N, K, G);

        hipMemcpy(d_act, h_act.data(), M * K / 2, hipMemcpyHostToDevice);
        hipMemcpy(d_wgt, h_wgt.data(), N * K / 2, hipMemcpyHostToDevice);
        for (int i = 0; i < M * K / G; i++)
                h_ascales_half[i] = __float2half(h_ascales[i]);
        for (int i = 0; i < N * K / G; i++)
                h_wscales_half[i] = __float2half(h_wscales[i]);
        hipMemcpy(d_ascales, h_ascales_half.data(), M * K / G * sizeof(__half),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_wscales, h_wscales_half.data(), N * K / G * sizeof(__half),
                  hipMemcpyHostToDevice);

        svdquant_scaled_mm_w4a4_kernel<__half, false, false, false, false>
            <<<dim3(1, 1), dim3(256), 0, 0>>>(d_act, d_wgt, d_ascales, d_wscales, nullptr, nullptr,
                                              nullptr, d_out, M, N, K, 0);
        hipDeviceSynchronize();
        hipMemcpy(h_out_gpu.data(), d_out, M * N * sizeof(__half), hipMemcpyHostToDevice);

        errors = 0;
        max_diff = 0;
        for (int i = 0; i < M * N && errors < 10; i++)
        {
                float gpu = __half2float(h_out_gpu[i]);
                float cpu = h_out_cpu[i];
                float diff = fabsf(gpu - cpu);
                if (diff > 1.0f)
                {
                        printf("  ERROR [%d,%d]: gpu=%.1f cpu=%.1f diff=%.1f\n", i / N, i % N, gpu,
                               cpu, diff);
                        errors++;
                }
                if (diff > max_diff) max_diff = diff;
        }
        // Test 3: Multiple groups
        printf("\nTest 3: Multiple groups (K=256, G=64, 4 groups)\n");
        constexpr int K2 = 256;
        constexpr int G2 = 64;
        constexpr int groups = K2 / G2;  // 4

        std::vector<int8_t> h_act2(M * K2 / 2);
        std::vector<int8_t> h_wgt2(N * K2 / 2);
        std::vector<float> h_ascales2(M * groups);
        std::vector<float> h_wscales2(N * groups);
        std::vector<__half> h_out_gpu2(M * N);
        std::vector<float> h_out_cpu2(M * N);
        std::vector<__half> h_ascales_half2(M * groups);
        std::vector<__half> h_wscales_half2(N * groups);

        // Fill with simple values
        memset(h_act2.data(), 0x11, M * K2 / 2);  // all 1s
        memset(h_wgt2.data(), 0x11, N * K2 / 2);  // all 1s
        for (int i = 0; i < M * groups; i++)
                h_ascales2[i] = 1.0f;
        for (int i = 0; i < N * groups; i++)
                h_wscales2[i] = 1.0f;

        // Expected: 4 groups * 64 pairs * 1*1 * 1*1 = 256 per output
        float expected = 256.0f;

        svdquant_mm_cpu(h_act2.data(), h_wgt2.data(), h_ascales2.data(), h_wscales2.data(),
                        h_out_cpu2.data(), M, N, K2, G2);
        printf("  CPU out[0,0] = %.1f (expected %.1f)\n", h_out_cpu2[0], expected);

        int8_t *d_act2, *d_wgt2;
        __half *d_ascales2, *d_wscales2, *d_out2;
        hipMalloc(&d_act2, M * K2 / 2);
        hipMalloc(&d_wgt2, N * K2 / 2);
        hipMalloc(&d_ascales2, M * groups * sizeof(__half));
        hipMalloc(&d_wscales2, N * groups * sizeof(__half));
        hipMalloc(&d_out2, M * N * sizeof(__half));

        hipMemcpy(d_act2, h_act2.data(), M * K2 / 2, hipMemcpyHostToDevice);
        hipMemcpy(d_wgt2, h_wgt2.data(), N * K2 / 2, hipMemcpyHostToDevice);
        for (int i = 0; i < M * groups; i++)
                h_ascales_half2[i] = __float2half(h_ascales2[i]);
        for (int i = 0; i < N * groups; i++)
                h_wscales_half2[i] = __float2half(h_wscales2[i]);
        hipMemcpy(d_ascales2, h_ascales_half2.data(), M * groups * sizeof(__half),
                  hipMemcpyHostToDevice);
        hipMemcpy(d_wscales2, h_wscales_half2.data(), N * groups * sizeof(__half),
                  hipMemcpyHostToDevice);

        dim3 grid2(1, 1);
        svdquant_scaled_mm_w4a4_kernel<__half, false, false, false, false>
            <<<grid2, dim3(256), 0, 0>>>(d_act2, d_wgt2, d_ascales2, d_wscales2, nullptr, nullptr,
                                         nullptr, d_out2, M, N, K2, 0);

        hipError_t err1 = hipDeviceSynchronize();
        if (err != hipSuccess)
        {
                printf("  GPU ERROR: %s\n", hipGetErrorString(err1));
        }
        else
        {
                hipMemcpy(h_out_gpu2.data(), d_out2, M * N * sizeof(__half), hipMemcpyHostToDevice);

                errors = 0;
                max_diff = 0;
                for (int i = 0; i < M * N && errors < 10; i++)
                {
                        float gpu = __half2float(h_out_gpu2[i]);
                        float cpu = h_out_cpu2[i];
                        float diff = fabsf(gpu - cpu);
                        if (diff > 1.0f)
                        {
                                printf("  ERROR [%d,%d]: gpu=%.1f cpu=%.1f\n", i / N, i % N, gpu,
                                       cpu);
                                errors++;
                        }
                        if (diff > max_diff) max_diff = diff;
                }
                if (errors == 0)
                        printf("  PASS (max diff = %.2f)\n", max_diff);
                else
                        printf("  FAIL: %d errors (max diff = %.2f)\n", errors, max_diff);
        }

        hipFree(d_act2);
        hipFree(d_wgt2);
        hipFree(d_ascales2);
        hipFree(d_wscales2);
        hipFree(d_out2);

        if (errors == 0)
                printf("  PASS (max diff = %.2f)\n", max_diff);
        else
                printf("  FAIL: %d errors (max diff = %.2f)\n", errors, max_diff);

        hipFree(d_act);
        hipFree(d_wgt);
        hipFree(d_ascales);
        hipFree(d_wscales);
        hipFree(d_out);
        return errors ? 1 : 0;
}
