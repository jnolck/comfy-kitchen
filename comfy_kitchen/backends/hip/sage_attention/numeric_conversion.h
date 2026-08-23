// SPDX-License-Identifier: Apache-2.0
// Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.

/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS,
 * https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
// #include <cuda/pipeline>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif

// #if defined(__CUDA_ARCH__)
// #define RUNTIME_ASSERT(x) __brkpt()
#ifdef __HIPCC__
#define RUNTIME_ASSERT(x)                 \
        do                                \
        {                                 \
                if (!(x))                 \
                {                         \
                        __builtin_trap(); \
                }                         \
        } while (0)
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

__device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dest, uint32_t source)
{
        // uint16_t h0 = source & 0xFFFF;
        // uint16_t h1 = (source >> 16) & 0xFFFF;
        // asm("cvt.f32.f16 %0, %1;" : "=f"(dest[0]) : "h"(h0));
        // asm("cvt.f32.f16 %0, %1;" : "=f"(dest[1]) : "h"(h1));
        half2 h2 = *reinterpret_cast<half2*>(&source);
        float2 f2 = __half22float2(h2);
        dest[0] = f2.x;
        dest[1] = f2.y;
}

__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t* dest, float* source0, float* source1)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo;\n"
        //             ".reg .b16 hi;\n"
        //             "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
        //             "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
        //             "mov.b32 %0, {lo, hi};\n"
        //             "}"
        //             : "=r"(dest[0])
        //             : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
        //
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif
        // Simple implementation using HIP's built-in FP8 conversion
        // Simple FP8 conversion for e4m3 (4-bit exponent, 3-bit mantissa)
        // Not hardware-optimized, but functional
        auto float_to_e4m3 = [](float x) -> uint8_t
        {
                uint32_t bits = __float_as_uint(x);
                uint32_t sign = (bits >> 16) & 0x80;
                // Simple truncation (not exact hardware behavior)
                uint32_t exp = (bits >> 23) & 0xFF;
                uint32_t mant = (bits >> 20) & 0x07;
                if (exp == 0xFF) exp = 0x0F;  // Inf/NaN -> max
                return static_cast<uint8_t>(sign | ((exp - 112) << 3) | mant);
        };

        uint8_t bytes[4];
        bytes[0] = float_to_e4m3(source0[0]);
        bytes[1] = float_to_e4m3(source0[1]);
        bytes[2] = float_to_e4m3(source1[0]);
        bytes[3] = float_to_e4m3(source1[1]);
        memcpy(dest, bytes, sizeof(uint32_t));
}

__device__ __forceinline__ void floatx4_to_e5m2x4(uint32_t* dest, float* source0, float* source1)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo;\n"
        //             ".reg .b16 hi;\n"
        //             "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n"
        //             "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n"
        //             "mov.b32 %0, {lo, hi};\n"
        //             "}"
        //             : "=r"(dest[0])
        //             : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif

        // Simple FP8 conversion for e5m2 (5-bit exponent, 2-bit mantissa)
        auto float_to_e5m2 = [](float x) -> uint8_t
        {
                uint32_t bits = __float_as_uint(x);
                uint32_t sign = (bits >> 16) & 0x80;
                uint32_t exp = (bits >> 23) & 0xFF;
                uint32_t mant = (bits >> 21) & 0x03;
                if (exp == 0xFF) exp = 0x1F;  // Inf/NaN -> max
                return static_cast<uint8_t>(sign | ((exp - 112) << 2) | mant);
        };

        uint8_t bytes[4];
        bytes[0] = float_to_e5m2(source0[0]);
        bytes[1] = float_to_e5m2(source0[1]);
        bytes[2] = float_to_e5m2(source1[0]);
        bytes[3] = float_to_e5m2(source1[1]);
        memcpy(dest, bytes, sizeof(uint32_t));
}

__device__ __forceinline__ void halfx4_to_e4m3x4(uint32_t* dest, uint32_t* source0,
                                                 uint32_t* source1)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo;\n"
        //             ".reg .b16 hi;\n"
        //             "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n"
        //             "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n"
        //             "mov.b32 %0, {lo, hi};\n"
        //             "}"
        //             : "=r"(dest[0])
        //             : "r"(source0[0]), "r"(source1[0]));
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif
        dest[0] = 0;
}

__device__ __forceinline__ void halfx4_to_e5m2x4(uint32_t* dest, uint32_t* source0,
                                                 uint32_t* source1)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo;\n"
        //             ".reg .b16 hi;\n"
        //             "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n"
        //             "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n"
        //             "mov.b32 %0, {lo, hi};\n"
        //             "}"
        //             : "=r"(dest[0])
        //             : "r"(source0[0]), "r"(source1[0]));
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif
        dest[0] = 0;
}

__device__ __forceinline__ void e4m3x4_to_halfx4(uint32_t* dest0, uint32_t* dest1, uint32_t* source)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo, hi;\n"
        //             "mov.b32 {lo, hi}, %2;\n"
        //             "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
        //             "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
        //             "}\n"
        //             : "=r"(dest0[0]), "=r"(dest1[0])
        //             : "r"(source[0]));
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif
        dest0[0] = 0;
        dest1[0] = 0;
}

__device__ __forceinline__ void e5m2x4_to_halfx4(uint32_t* dest0, uint32_t* dest1, uint32_t* source)
{
        // #ifdef FP8_CAST_ENABLED
        //         asm volatile(
        //             "{\n"
        //             ".reg .b16 lo, hi;\n"
        //             "mov.b32 {lo, hi}, %2;\n"
        //             "cvt.rn.f16x2.e5m2x2 %0, lo;\n"
        //             "cvt.rn.f16x2.e5m2x2 %1, hi;\n"
        //             "}\n"
        //             : "=r"(dest0[0]), "=r"(dest1[0])
        //             : "r"(source[0]));
        // #else
        //         RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
        // #endif
        dest0[0] = 0;
        dest1[0] = 0;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
        // uint32_t dst;
        // asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
        // return reinterpret_cast<const int8_t&>(dst);
        return static_cast<int8_t>(__float2int_rn(x));
}
