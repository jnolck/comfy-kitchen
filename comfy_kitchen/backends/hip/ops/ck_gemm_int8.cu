/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ROCm/HIP ck_tile: INT8 GEMM with FUSED dequant for RDNA3/4
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n] -> out
 * dtype
 *
 * Uses ck_tile's GemmKernelMultiD with broadcast + CDE element-wise epilogue.
 * Based on examples 19_gemm_multi_d and 22_gemm_multi_abd.
 */
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define COMFY_HAVE_CK

#ifdef COMFY_HAVE_CK

#include <map>
#include <mutex>
#include <tuple>

#include "../utils.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace comfy
{

// ============================================================================
// Dequant epilogue: D = acc * ds0 * ds1 + ds2  (per-element multiply-add)
// ============================================================================
struct FusedDequantEpilogue
{
        template <typename E, typename C, typename D0, typename D1, typename D2>
        CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const D0& d0, const D1& d1,
                                            const D2& d2) const
        {
                float val = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                                ck_tile::type_convert<float>(d1) +
                            ck_tile::type_convert<float>(d2);
                e = ck_tile::type_convert<E>(val);
        }
};

// ============================================================================
// GemmConfig for RDNA3 WMMA int8
// ============================================================================
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int WTM, int WTN, int WTK,
          int BlockPerCu>
struct GemmConfigInt8WMMA
{
        static constexpr ck_tile::index_t M_Tile = TBM;
        static constexpr ck_tile::index_t N_Tile = TBN;
        static constexpr ck_tile::index_t K_Tile = TBK;
        static constexpr ck_tile::index_t M_Warp = WM;
        static constexpr ck_tile::index_t N_Warp = WN;
        static constexpr ck_tile::index_t K_Warp = WK;
        static constexpr ck_tile::index_t M_Warp_Tile = WTM;
        static constexpr ck_tile::index_t N_Warp_Tile = WTN;
        static constexpr ck_tile::index_t K_Warp_Tile = WTK;
        static constexpr int kBlockPerCu = BlockPerCu;

        static constexpr bool kPadM = true;
        static constexpr bool kPadN = true;
        static constexpr bool kPadK = true;
        static constexpr bool PermuteA = false;
        static constexpr bool PermuteB = false;
        static constexpr bool TransposeC = false;
        static constexpr bool UseStructuredSparsity = false;

        // static constexpr int kBlockPerCu = 2;
        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01 = 4;
        static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
        static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
        static constexpr bool DoubleSmemBuffer = false;
};

// ============================================================================
// Broadcast helpers
// ============================================================================
__global__ void broadcast_row(const float* src, float* dst, int M, int N)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N) return;
        dst[idx] = src[idx / N];
}

__global__ void broadcast_col(const float* src, float* dst, int M, int N)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N) return;
        dst[idx] = src[idx % N];
}

// ============================================================================
// One fused int8 GEMM instance
// ============================================================================
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int WTM,
          int WTN, int WTK, int kBlockPerCu>
struct FusedInt8GemmCKTile
{
        using ADataType = ck_tile::int8_t;
        using BDataType = ck_tile::int8_t;
        using AccDataType = int32_t;
        using CDataType = ElementOutput;

        using D0DataType = float;
        using D1DataType = float;
        using D2DataType = float;
        using DsDataType = ck_tile::tuple<D0DataType, D1DataType, D2DataType>;

        using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
        using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
        using ELayout = ck_tile::tensor_layout::gemm::RowMajor;
        using DsLayout = ck_tile::tuple<ELayout, ELayout, ELayout>;  // all RowMajor [M,N]

        using GemmConfig =
            GemmConfigInt8WMMA<TBM, TBN, TBK, WM, WN, WK, WTM, WTN, WTK, kBlockPerCu>;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile,
                              GemmConfig::K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape, GemmConfig::TileParitionerGroupNum, GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM, GemmConfig::kPadN,
                                             GemmConfig::kPadK, GemmConfig::DoubleSmemBuffer,
                                             ALayout, BLayout, ELayout, GemmConfig::TransposeC>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape,
                                                  GemmUniversalTraits, GemmConfig::Scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            ADataType, BDataType, DsDataType, AccDataType, CDataType, DsLayout, ELayout,
            FusedDequantEpilogue, TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::M_Warp_Tile,
            GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile, UniversalGemmProblem::TransposeC>>;

        using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;

        static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                        const float* bias, ElementOutput* D, int M, int N, int K,
                        hipStream_t stream)
        {
                // printf("CK run: M=%d N=%d K=%d TBM=%d TBN=%d TBK=%d\n", M, N, K, TBM,
                // TBN, TBK);
                float *d_xs_full, *d_ws_full, *d_bias_full;
                CUDA_CHECK(hipMalloc(&d_xs_full, M * N * sizeof(float)));
                CUDA_CHECK(hipMalloc(&d_ws_full, M * N * sizeof(float)));
                CUDA_CHECK(hipMalloc(&d_bias_full, M * N * sizeof(float)));

                int bcast_threads = 256;
                int bcast_blocks = (M * N + bcast_threads - 1) / bcast_threads;

                broadcast_row<<<bcast_blocks, bcast_threads, 0, stream>>>(xs, d_xs_full, M, N);
                broadcast_col<<<bcast_blocks, bcast_threads, 0, stream>>>(ws, d_ws_full, M, N);
                if (bias == nullptr)
                {
                        CUDA_CHECK(hipMemsetAsync(d_bias_full, 0, M * N * sizeof(float), stream));
                }
                else
                {
                        broadcast_col<<<bcast_blocks, bcast_threads, 0, stream>>>(bias, d_bias_full,
                                                                                  M, N);
                }

                CUDA_CHECK(hipStreamSynchronize(stream));
                auto err = hipGetLastError();
                if (err != hipSuccess)
                {
                        CUDA_CHECK(hipFree(d_xs_full));
                        CUDA_CHECK(hipFree(d_ws_full));
                        CUDA_CHECK(hipFree(d_bias_full));
                        printf("  broadcast failed: %s\n", hipGetErrorString(err));
                        return false;
                }
                // printf("  broadcast OK\n");

                using GemmMultiDArgs = ck_tile::GemmMultiDHostArgs<DsDataType::size()>;

                int stride_A = K, stride_B = K, stride_E = N;
                int stride_D = N;  // RowMajor [M,N]

                GemmMultiDArgs args = {const_cast<int8_t*>(A),
                                       const_cast<int8_t*>(B),
                                       {d_xs_full, d_ws_full, d_bias_full},
                                       D,
                                       1,
                                       M,
                                       N,
                                       K,
                                       stride_A,
                                       stride_B,
                                       {stride_D, stride_D, stride_D},
                                       stride_E};

                auto kargs = Kernel::MakeKernelArgs(args);

                if (!Kernel::IsSupportedArgument(kargs))
                {
                        CUDA_CHECK(hipFree(d_xs_full));
                        CUDA_CHECK(hipFree(d_ws_full));
                        CUDA_CHECK(hipFree(d_bias_full));
                        return false;
                }

                const dim3 grids = Kernel::GridSize(M, N, 1);
                const dim3 blocks = Kernel::BlockSize();
                // printf("  grid=(%d,%d,%d) block=(%d,%d,%d)\n", grids.x, grids.y, grids.z,
                // blocks.x, blocks.y, blocks.z);

                ck_tile::stream_config s{stream, false, 1};
                float elapsed =
                    ck_tile::launch_kernel(s, ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                                  Kernel{}, grids, blocks, 0, kargs));

                // CUDA_CHECK(hipStreamSynchronize(stream));
                err = hipGetLastError();
                if (err != hipSuccess)
                {
                        CUDA_CHECK(hipFree(d_xs_full));
                        CUDA_CHECK(hipFree(d_ws_full));
                        CUDA_CHECK(hipFree(d_bias_full));
                        printf("  kernel error: %s\n", hipGetErrorString(err));
                        return false;
                }
                CUDA_CHECK(hipFree(d_xs_full));
                CUDA_CHECK(hipFree(d_ws_full));
                CUDA_CHECK(hipFree(d_bias_full));

                return elapsed >= 0;
        }
};

// ============================================================================
// Autotuning dispatcher
// ============================================================================

template <typename OutT>
bool dispatch_fused_ck(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                       const float* bias, OutT* D, int M, int N, int K, hipStream_t stream)
{
        using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*,
                            OutT*, int, int, int, hipStream_t);

        static const Fn runners[] = {
            // just commenting this out to improve compile times
            &FusedInt8GemmCKTile<OutT, 128, 128, 64, 4, 2, 1, 16, 16, 16, 1>::run,
            &FusedInt8GemmCKTile<OutT, 128, 256, 64, 4, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTile<OutT, 256, 128, 64, 4, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTile<OutT, 64, 64, 32, 2, 2, 1, 16, 16, 16, 1>::run,
            &FusedInt8GemmCKTile<OutT, 64, 128, 32, 2, 2, 1, 16, 16, 16, 1>::run,
            &FusedInt8GemmCKTile<OutT, 128, 64, 32, 2, 2, 1, 16, 16, 16, 2>::run,
            &FusedInt8GemmCKTile<OutT, 32, 64, 32, 1, 2, 1, 16, 16, 16, 1>::run,
        };
        constexpr int NC = sizeof(runners) / sizeof(runners[0]);

        static std::mutex mtx;
        static std::map<std::tuple<int, int, int>, int> cache;
        const std::tuple<int, int, int> key{M, N, K};

        int best;
        {
                std::lock_guard<std::mutex> lk(mtx);
                auto it = cache.find(key);
                best = (it != cache.end()) ? it->second : -2;
        }

        if (best == -2)
        {
                best = -1;
                float best_ms = 1e30f;
                hipEvent_t s, e;
                CUDA_CHECK(hipEventCreate(&s));
                CUDA_CHECK(hipEventCreate(&e));

                for (int i = 0; i < NC; ++i)
                {
                        if (!runners[i](A, B, xs, ws, bias, D, M, N, K, stream)) continue;

                        // CUDA_CHECK(hipStreamSynchronize(stream));
                        CUDA_CHECK(hipEventRecord(s, stream));
                        for (int r = 0; r < 3; ++r)
                        {
                                runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
                                CUDA_CHECK(hipStreamSynchronize(stream));
                        }
                        // Clean this up, it's just to get rid of the compile warnings
                        (void)hipEventRecord(e, stream);
                        (void)hipEventSynchronize(e);

                        float ms = 0.f;
                        (void)hipEventElapsedTime(&ms, s, e);
                        if (ms < best_ms)
                        {
                                best_ms = ms;
                                best = i;
                        }
                }

                (void)hipEventDestroy(s);
                (void)hipEventDestroy(e);

                std::lock_guard<std::mutex> lk(mtx);
                cache[key] = best;
        }

        if (best < 0) return false;
        return runners[best](A, B, xs, ws, bias, D, M, N, K, stream);
}

}  // namespace comfy

extern "C"
{
        bool launch_cutlass_int8_dequant(const void* A, const void* B, const void* xs,
                                         const void* ws, const void* bias, void* D, int64_t M,
                                         int64_t N, int64_t K, int out_dtype_code,
                                         hipStream_t stream)
        {
                try
                {
                        if (M == 0 || N == 0 || K == 0) return true;

                        const int8_t* a = static_cast<const int8_t*>(A);
                        const int8_t* b = static_cast<const int8_t*>(B);
                        const float* x = static_cast<const float*>(xs);
                        const float* w = static_cast<const float*>(ws);
                        const float* bs = static_cast<const float*>(bias);

                        switch (out_dtype_code)
                        {
                                case 0:
                                        return comfy::dispatch_fused_ck<float>(
                                            a, b, x, w, bs, (float*)D, M, N, K, stream);
                                case 1:
                                        return comfy::dispatch_fused_ck<ck_tile::half_t>(
                                            a, b, x, w, bs, (ck_tile::half_t*)D, M, N, K, stream);
                                case 2:
                                        return comfy::dispatch_fused_ck<ck_tile::bf16_t>(
                                            a, b, x, w, bs, (ck_tile::bf16_t*)D, M, N, K, stream);
                                default:
                                        return false;
                        }
                }
                catch (const std::exception& e)
                {
                        fprintf(stderr, "CK EXCEPTION: %s\n", e.what());
                        return false;
                }
                catch (...)
                {
                        fprintf(stderr, "CK UNKNOWN EXCEPTION\n");
                        return false;
                }
        }

}  // extern "C"

#endif  // COMFY_HAVE_CK
