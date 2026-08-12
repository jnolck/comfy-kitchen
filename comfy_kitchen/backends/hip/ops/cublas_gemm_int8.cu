/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <hipblaslt.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../dtype_dispatch.h"
#include "../hipblaslt_runtime.h"
#include "utils.h"

// Helper macro for cuBLAS error checking
#define CUBLAS_CHECK(call)                                                       \
        do                                                                       \
        {                                                                        \
                hipblasStatus_t status = call;                                   \
                if (status != HIPBLAS_STATUS_SUCCESS)                            \
                {                                                                \
                        throw std::runtime_error(std::string("cuBLAS error: ") + \
                                                 std::to_string(status));        \
                }                                                                \
        } while (0)

namespace comfy
{

namespace
{

// Thread-local handle cache to avoid creating/destroying handles repeatedly.
thread_local hipblasLtHandle_t cached_int8_handle = nullptr;

hipblasLtHandle_t get_cublas_lt_handle_int8()
{
        auto& runtime = HipblasLtRuntime::instance();
        if (!runtime.is_available())
        {
                throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
        }

        if (cached_int8_handle == nullptr)
        {
                hipblasStatus_t status = runtime.hipblasLtCreate(&cached_int8_handle);
                if (status != HIPBLAS_STATUS_SUCCESS)
                {
                        throw std::runtime_error(std::string("cuBLAS handle creation error: ") +
                                                 std::to_string(status));
                }
        }
        return cached_int8_handle;
}

struct Int8GemmKey
{
        int64_t M;
        int64_t N;
        int64_t K;
        int64_t workspace_size;

        bool operator==(const Int8GemmKey& other) const
        {
                return M == other.M && N == other.N && K == other.K &&
                       workspace_size == other.workspace_size;
        }
};

struct Int8GemmKeyHash
{
        std::size_t operator()(const Int8GemmKey& key) const
        {
                std::size_t h = static_cast<std::size_t>(key.M);
                h ^= static_cast<std::size_t>(key.N) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= static_cast<std::size_t>(key.K) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= static_cast<std::size_t>(key.workspace_size) + 0x9e3779b97f4a7c15ULL +
                     (h << 6) + (h >> 2);
                return h;
        }
};

struct CachedInt8GemmPlan
{
        hipblasLtMatmulDesc_t operationDesc = nullptr;
        hipblasLtMatrixLayout_t Adesc = nullptr;
        hipblasLtMatrixLayout_t Bdesc = nullptr;
        hipblasLtMatrixLayout_t Cdesc = nullptr;
        hipblasLtMatmulAlgo_t algo = {};
        size_t workspace_size = 0;
};

thread_local std::unordered_map<Int8GemmKey, CachedInt8GemmPlan, Int8GemmKeyHash> cached_int8_plans;

CachedInt8GemmPlan& get_int8_gemm_plan(hipblasLtHandle_t ltHandle, int64_t M, int64_t N, int64_t K,
                                       int64_t workspace_size)
{
        auto& runtime = HipblasLtRuntime::instance();
        const Int8GemmKey key{M, N, K, workspace_size > 0 ? workspace_size : 0};
        auto found = cached_int8_plans.find(key);
        if (found != cached_int8_plans.end())
        {
                return found->second;
        }

        CachedInt8GemmPlan plan;
        plan.workspace_size = static_cast<size_t>(key.workspace_size);

        CUBLAS_CHECK(
            runtime.hipblasLtMatmulDescCreate(&plan.operationDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I));

        // In cuBLAS column-major terms, we want to compute C_col = B_row @ A_row^T.
        // B_row in memory is [N, K] row-major -> [K, N] col-major and is transposed.
        // A_row in memory is [M, K] row-major -> [K, M] col-major and is not
        // transposed.
        const hipblasOperation_t transa = HIPBLAS_OP_T;
        const hipblasOperation_t transb = HIPBLAS_OP_N;
        CUBLAS_CHECK(runtime.hipblasLtMatmulDescSetAttribute(
            plan.operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CUBLAS_CHECK(runtime.hipblasLtMatmulDescSetAttribute(
            plan.operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        CUBLAS_CHECK(runtime.hipblasLtMatrixLayoutCreate(&plan.Bdesc, HIP_R_8I, K, N, K));
        CUBLAS_CHECK(runtime.hipblasLtMatrixLayoutCreate(&plan.Adesc, HIP_R_8I, K, M, K));
        CUBLAS_CHECK(runtime.hipblasLtMatrixLayoutCreate(&plan.Cdesc, HIP_R_32I, N, M, N));

        hipblasLtMatmulPreference_t preference = nullptr;
        CUBLAS_CHECK(runtime.hipblasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(runtime.hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &plan.workspace_size,
            sizeof(plan.workspace_size)));

        hipblasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedResults = 0;
        const auto status = runtime.hipblasLtMatmulAlgoGetHeuristic(
            ltHandle, plan.operationDesc, plan.Bdesc, plan.Adesc, plan.Cdesc, plan.Cdesc,
            preference, 1, &heuristicResult, &returnedResults);

        CUBLAS_CHECK(runtime.hipblasLtMatmulPreferenceDestroy(preference));

        if (status == HIPBLAS_STATUS_NOT_SUPPORTED || returnedResults == 0)
        {
                if (plan.Cdesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Cdesc);
                if (plan.Bdesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Bdesc);
                if (plan.Adesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Adesc);
                if (plan.operationDesc) runtime.hipblasLtMatmulDescDestroy(plan.operationDesc);
                throw std::runtime_error(
                    "INT8 GEMM not supported on this GPU for these dimensions (requires SM "
                    ">= 7.5, and dimensions multiple of 4).");
        }
        if (status != HIPBLAS_STATUS_SUCCESS)
        {
                if (plan.Cdesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Cdesc);
                if (plan.Bdesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Bdesc);
                if (plan.Adesc) runtime.hipblasLtMatrixLayoutDestroy(plan.Adesc);
                if (plan.operationDesc) runtime.hipblasLtMatmulDescDestroy(plan.operationDesc);
                throw std::runtime_error(std::string("cuBLAS heuristic error: ") +
                                         std::to_string(status));
        }

        plan.algo = heuristicResult.algo;
        auto inserted = cached_int8_plans.emplace(key, plan);
        return inserted.first->second;
}

void cublas_gemm_int8_impl(const int8_t* A_ptr,  // [M, K] row-major
                           const int8_t* B_ptr,  // [N, K] row-major
                           int32_t* C_ptr,       // [M, N] row-major output
                           int64_t M, int64_t N, int64_t K, void* workspace_ptr,
                           int64_t workspace_size, hipStream_t stream)
{
        auto& runtime = HipblasLtRuntime::instance();
        if (!runtime.is_available())
        {
                throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
        }

        if (M == 0 || N == 0 || K == 0)
        {
                return;
        }

        hipblasLtHandle_t ltHandle = get_cublas_lt_handle_int8();
        CachedInt8GemmPlan& plan = get_int8_gemm_plan(ltHandle, M, N, K, workspace_size);

        // Alpha and beta scalars
        int32_t alpha = 1;
        int32_t beta = 0;

        // Execute matmul
        CUBLAS_CHECK(runtime.hipblasLtMatmul(ltHandle, plan.operationDesc, &alpha,
                                             B_ptr,  // First operand
                                             plan.Bdesc,
                                             A_ptr,  // Second operand
                                             plan.Adesc, &beta, C_ptr, plan.Cdesc, C_ptr,
                                             plan.Cdesc, &plan.algo, workspace_ptr,
                                             plan.workspace_size, stream));
}

}  // anonymous namespace

}  // namespace comfy

// C interface for DLPack bindings
extern "C"
{
        void launch_cublas_gemm_int8_kernel(const void* A_ptr, const void* B_ptr, void* C_ptr,
                                            int64_t M, int64_t N, int64_t K, void* workspace_ptr,
                                            int64_t workspace_size, hipStream_t stream)
        {
                comfy::cublas_gemm_int8_impl(
                    static_cast<const int8_t*>(A_ptr), static_cast<const int8_t*>(B_ptr),
                    static_cast<int32_t*>(C_ptr), M, N, K, workspace_ptr, workspace_size, stream);
        }

}  // extern "C"
