# File: tests/comprehensive_fp8_test.py

import torch
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import itertools
import warnings

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)


class FP8TestSuite:
    def __init__(self, device="cuda", verbose=True):
        self.device = device
        self.verbose = verbose
        self.stream_ptr = torch.cuda.current_stream().cuda_stream
        self.passed = 0
        self.failed = 0
        self.errors = []

        # Dtype codes matching your C interface
        self.DTYPE_F32 = 0
        self.DTYPE_F16 = 1
        self.DTYPE_BF16 = 2
        self.DTYPE_F8_E4M3 = 5
        self.DTYPE_F8_E5M2 = 6

        # FP8 format properties
        self.E4M3_MAX = 448.0
        self.E4M3_MIN_NORMAL = 2**-6  # 0.015625
        self.E4M3_MIN_SUBNORMAL = 2**-9  # ~0.00195
        self.E5M2_MAX = 57344.0

    def log(self, msg, force=False):
        if self.verbose or force:
            print(msg)

    def check_result(self, test_name: str, success: bool, details: str = ""):
        if success:
            self.passed += 1
            self.log(f"✓ {test_name}")
        else:
            self.failed += 1
            self.errors.append((test_name, details))
            self.log(f"✗ {test_name} FAILED: {details}")
        return success

    # =========================================================================
    # Test 1: Basic Quantize-Dequantize Roundtrip
    # =========================================================================
    def test_basic_roundtrip(self):
        self.log("\n=== Test 1: Basic Quantize-Dequantize Roundtrip ===")

        sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
        all_pass = True

        for size in sizes:
            # Create test data
            x = torch.linspace(
                -self.E4M3_MAX * 0.9,
                self.E4M3_MAX * 0.9,
                size,
                device=self.device,
                dtype=torch.float16,
            )
            scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)

            # Quantize using PyTorch's built-in (reference)
            x_fp8 = x.to(torch.float8_e4m3fnuz)

            # Dequantize using our kernel
            output = torch.zeros(size, device=self.device, dtype=torch.float16)
            _C.dequantize_per_tensor_fp8(
                x_fp8, scale, output, self.DTYPE_F8_E4M3, self.DTYPE_F16, x.numel(), self.stream_ptr
            )
            torch.cuda.synchronize()

            # Compare
            expected = x_fp8.to(torch.float16)
            max_diff = (output - expected).abs().max().item()

            success = max_diff < 0.01  # Should be exact since same format
            all_pass &= self.check_result(
                f"size={size:4d}", success, f"Max diff: {max_diff}" if not success else ""
            )

            if not success and self.verbose:
                # Show first few mismatches
                mismatches = (output != expected).nonzero(as_tuple=True)[0]
                for idx in mismatches[:5]:
                    i = idx.item()
                    self.log(
                        f"  [{i}] got={output[i].item():.4f} expected={expected[i].item():.4f}"
                    )

        return all_pass

    # =========================================================================
    # Test 2: Special Values (Zero, NaN, Inf, Denormals)
    # =========================================================================
    def test_special_values(self):
        self.log("\n=== Test 2: Special Values ===")

        # Create tensor with special values
        special_values = torch.tensor(
            [
                0.0,
                -0.0,
                float("inf"),
                float("-inf"),
                float("nan"),
                1.401298464324817e-45,  # Smallest positive float32
                -1.401298464324817e-45,
                1.175494350822288e-38,  # Smallest normalized float32
                -1.175494350822288e-38,
            ],
            device=self.device,
            dtype=torch.float16,
        )

        # Also test Python float special values
        special_floats = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
        for f in special_floats:
            try:
                t = torch.tensor([f], device=self.device, dtype=torch.float16)
                special_values = torch.cat([special_values, t])
            except:
                pass

        scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
        numel = special_values.numel()

        # Quantize and check for NaN in output
        x_fp8 = torch.zeros(numel, device=self.device, dtype=torch.float8_e4m3fnuz)

        # Test quantization kernel
        try:
            _C.quantize_per_tensor_fp8(
                special_values,
                scale,
                x_fp8,
                self.DTYPE_F16,
                self.DTYPE_F8_E4M3,
                numel,
                self.stream_ptr,
            )
            torch.cuda.synchronize()

            # Check for NaN in FP8 output
            fp8_bytes = x_fp8.view(torch.uint8)
            nan_mask = self._detect_fp8_nan(fp8_bytes)
            nan_count = nan_mask.sum().item()

            self.check_result(
                f"Special values: NaN count = {nan_count}",
                nan_count == 0,
                f"Found {nan_count} NaN values in output" if nan_count > 0 else "",
            )

            if nan_count > 0 and self.verbose:
                nan_indices = nan_mask.nonzero(as_tuple=True)[0]
                for idx in nan_indices[:10]:
                    i = idx.item()
                    self.log(
                        f"  [{i}] input={special_values[i].item():.6e} "
                        f"-> fp8=0x{fp8_bytes[i].item():02x} (NaN)"
                    )

        except Exception as e:
            self.check_result("Special values", False, f"Exception: {e}")

    # =========================================================================
    # Test 3: Scale Testing (Overflow, Underflow, Edge Cases)
    # =========================================================================
    def test_scale_variations(self):
        self.log("\n=== Test 3: Scale Variations ===")

        # Test values within FP8 range
        values = torch.linspace(-10.0, 10.0, 128, device=self.device, dtype=torch.float16)

        scales = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6]

        for scale_val in scales:
            scale = torch.tensor([scale_val], device=self.device, dtype=torch.float32)
            x_fp8 = torch.zeros(128, device=self.device, dtype=torch.float8_e4m3fnuz)

            try:
                _C.quantize_per_tensor_fp8(
                    values, scale, x_fp8, self.DTYPE_F16, self.DTYPE_F8_E4M3, 128, self.stream_ptr
                )
                torch.cuda.synchronize()

                # Check for NaN
                fp8_bytes = x_fp8.view(torch.uint8)
                nan_count = self._detect_fp8_nan(fp8_bytes).sum().item()

                # For extreme scales, values should clamp (no NaN)
                success = nan_count == 0
                self.check_result(
                    f"scale={scale_val:8.1e}",
                    success,
                    f"NaN count: {nan_count}" if not success else "",
                )

            except Exception as e:
                self.check_result(f"scale={scale_val:8.1e}", False, f"Exception: {e}")

    # =========================================================================
    # Test 4: Large Tensor Processing
    # =========================================================================
    def test_large_tensors(self):
        self.log("\n=== Test 4: Large Tensor Processing ===")

        sizes = [1024, 4096, 16384, 65536]

        for size in sizes:
            # Generate random data with various distributions
            if size <= 4096:
                # Small enough to test multiple distributions
                x = torch.empty(size, device=self.device, dtype=torch.float16)
                quarter = size // 4
                x[:quarter].normal_(0, 1)
                x[quarter : 2 * quarter].uniform_(-10, 10)
                x[2 * quarter : 3 * quarter].exponential_(1.0)
                x[3 * quarter :] = torch.linspace(
                    -100, 100, size - 3 * quarter, device=self.device, dtype=torch.float16
                )
            else:
                x = torch.randn(size, device=self.device, dtype=torch.float16) * 10

            scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
            x_fp8 = torch.zeros(size, device=self.device, dtype=torch.float8_e4m3fnuz)

            try:
                _C.quantize_per_tensor_fp8(
                    x, scale, x_fp8, self.DTYPE_F16, self.DTYPE_F8_E4M3, size, self.stream_ptr
                )
                torch.cuda.synchronize()

                # Verify no NaN
                fp8_bytes = x_fp8.view(torch.uint8)
                nan_count = self._detect_fp8_nan(fp8_bytes).sum().item()

                # Verify against PyTorch's quantization
                expected_fp8 = x.to(torch.float8_e4m3fnuz)
                expected_bytes = expected_fp8.view(torch.uint8)

                # Exact match expected (same input, same scale)
                matches = (fp8_bytes == expected_bytes).sum().item()
                match_rate = matches / size

                success = (nan_count == 0) and (match_rate > 0.999)
                self.check_result(
                    f"size={size:6d}",
                    success,
                    f"NaN: {nan_count}, Match rate: {match_rate:.6f}" if not success else "",
                )

            except Exception as e:
                self.check_result(f"size={size:6d}", False, f"Exception: {e}")

    # =========================================================================
    # Test 5: Alignment/Boundary Conditions
    # =========================================================================
    def test_alignment(self):
        self.log("\n=== Test 5: Alignment & Boundary Conditions ===")

        # Test sizes that challenge kE4M3Alignment (16) alignment
        sizes = list(range(1, 33)) + [63, 64, 65, 127, 128, 129, 255, 256, 257]

        for size in sizes:
            x = torch.arange(size, device=self.device, dtype=torch.float16) % 10
            scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
            x_fp8 = torch.zeros(size, device=self.device, dtype=torch.float8_e4m3fnuz)

            try:
                _C.quantize_per_tensor_fp8(
                    x, scale, x_fp8, self.DTYPE_F16, self.DTYPE_F8_E4M3, size, self.stream_ptr
                )
                torch.cuda.synchronize()

                fp8_bytes = x_fp8.view(torch.uint8)
                nan_count = self._detect_fp8_nan(fp8_bytes).sum().item()

                success = nan_count == 0
                if not success:
                    self.check_result(f"size={size:3d}", False, f"NaN count: {nan_count}")

            except Exception as e:
                self.check_result(f"size={size:3d}", False, f"Exception: {e}")

        # Just report summary for alignment tests
        self.check_result("All alignment sizes", True)

    # =========================================================================
    # Test 6: Per-Tensor Quantization (Multiple Tensors)
    # =========================================================================
    def test_per_tensor_multiple(self):
        self.log("\n=== Test 6: Multiple Tensor Quantization ===")

        num_tensors = 8
        tensor_size = 512

        for t in range(num_tensors):
            # Different distributions per tensor
            x = torch.randn(tensor_size, device=self.device, dtype=torch.float16) * (t + 1)
            scale = torch.tensor([1.0 / (t + 1)], device=self.device, dtype=torch.float32)
            x_fp8 = torch.zeros(tensor_size, device=self.device, dtype=torch.float8_e4m3fnuz)

            try:
                _C.quantize_per_tensor_fp8(
                    x,
                    scale,
                    x_fp8,
                    self.DTYPE_F16,
                    self.DTYPE_F8_E4M3,
                    tensor_size,
                    self.stream_ptr,
                )
                torch.cuda.synchronize()

                fp8_bytes = x_fp8.view(torch.uint8)
                nan_count = self._detect_fp8_nan(fp8_bytes).sum().item()

                success = nan_count == 0
                self.check_result(
                    f"tensor {t} (scale={1 / (t + 1):.3f})",
                    success,
                    f"NaN count: {nan_count}" if not success else "",
                )

            except Exception as e:
                self.check_result(f"tensor {t}", False, f"Exception: {e}")

    # =========================================================================
    # Test 7: Dequantization Accuracy
    # =========================================================================
    def test_dequantize_accuracy(self):
        self.log("\n=== Test 7: Dequantization Accuracy ===")

        sizes = [256, 1024, 4096]

        for size in sizes:
            x = torch.randn(size, device=self.device, dtype=torch.float16) * 100
            scale_vals = [0.1, 1.0, 10.0]

            for scale_val in scale_vals:
                scale = torch.tensor([scale_val], device=self.device, dtype=torch.float32)

                # Quantize with PyTorch
                x_scaled = (x / scale_val).clamp(-self.E4M3_MAX, self.E4M3_MAX)
                x_fp8 = x_scaled.to(torch.float8_e4m3fnuz)

                # Dequantize with our kernel
                output = torch.zeros(size, device=self.device, dtype=torch.float16)
                _C.dequantize_per_tensor_fp8(
                    x_fp8, scale, output, self.DTYPE_F8_E4M3, self.DTYPE_F16, size, self.stream_ptr
                )
                torch.cuda.synchronize()

                # Compare
                expected = x_fp8.to(torch.float16) * scale_val
                abs_diff = (output - expected).abs()
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()

                # Check for NaN in output
                nan_mask = torch.isnan(output)
                nan_count = nan_mask.sum().item()

                success = (nan_count == 0) and (max_diff < 0.01)
                self.check_result(
                    f"size={size} scale={scale_val:5.1f}",
                    success,
                    f"NaN: {nan_count}, Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
                    if not success
                    else "",
                )

    # =========================================================================
    # Test 8: Numerical Stability (NaN Propagation)
    # =========================================================================
    def test_nan_propagation(self):
        self.log("\n=== Test 8: NaN Propagation Test ===")

        # Test specific patterns that might cause NaN
        test_cases = [
            ("Zero input", torch.zeros(16, device=self.device, dtype=torch.float16)),
            (
                "Negative values",
                torch.tensor([-1.0, -2.0, -4.0, -8.0], device=self.device, dtype=torch.float16),
            ),
            (
                "Large negative",
                torch.tensor([-448.0, -449.0, -500.0], device=self.device, dtype=torch.float16),
            ),
            (
                "Boundary values",
                torch.tensor(
                    [self.E4M3_MAX, -self.E4M3_MAX, self.E4M3_MAX + 1, -self.E4M3_MAX - 1],
                    device=self.device,
                    dtype=torch.float16,
                ),
            ),
            (
                "Powers of two",
                torch.tensor(
                    [2.0**i for i in range(-10, 10)], device=self.device, dtype=torch.float16
                ),
            ),
        ]

        for name, tensor in test_cases:
            scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
            x_fp8 = torch.zeros(tensor.numel(), device=self.device, dtype=torch.float8_e4m3fnuz)

            try:
                _C.quantize_per_tensor_fp8(
                    tensor,
                    scale,
                    x_fp8,
                    self.DTYPE_F16,
                    self.DTYPE_F8_E4M3,
                    tensor.numel(),
                    self.stream_ptr,
                )
                torch.cuda.synchronize()

                fp8_bytes = x_fp8.view(torch.uint8)
                nan_count = self._detect_fp8_nan(fp8_bytes).sum().item()

                success = nan_count == 0
                self.check_result(name, success, f"NaN count: {nan_count}" if not success else "")

                if not success and self.verbose:
                    for i in range(tensor.numel()):
                        if self._is_fp8_nan_single(fp8_bytes[i].item()):
                            self.log(
                                f"  Input: {tensor[i].item():.6e} -> FP8: 0x{fp8_bytes[i].item():02x}"
                            )

            except Exception as e:
                self.check_result(name, False, f"Exception: {e}")

    # =========================================================================
    # Test 9: Roundtrip Error Analysis
    # =========================================================================
    def test_roundtrip_error_analysis(self):
        self.log("\n=== Test 9: Roundtrip Error Analysis ===")

        size = 4096
        x = torch.randn(size, device=self.device, dtype=torch.float16) * 10
        scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)

        # Quantize
        x_fp8 = torch.zeros(size, device=self.device, dtype=torch.float8_e4m3fnuz)
        _C.quantize_per_tensor_fp8(
            x, scale, x_fp8, self.DTYPE_F16, self.DTYPE_F8_E4M3, size, self.stream_ptr
        )

        # Dequantize
        output = torch.zeros(size, device=self.device, dtype=torch.float16)
        _C.dequantize_per_tensor_fp8(
            x_fp8, scale, output, self.DTYPE_F8_E4M3, self.DTYPE_F16, size, self.stream_ptr
        )
        torch.cuda.synchronize()

        # Error analysis
        abs_error = (output - x).abs()
        rel_error = abs_error / (x.abs() + 1e-8)

        # Remove NaN/Inf for statistics
        valid_mask = ~(torch.isnan(output) | torch.isinf(output) | torch.isnan(x) | torch.isinf(x))

        if valid_mask.sum() > 0:
            valid_abs = abs_error[valid_mask]
            valid_rel = rel_error[valid_mask]

            self.log(f"  Max absolute error: {valid_abs.max().item():.6e}")
            self.log(f"  Mean absolute error: {valid_abs.mean().item():.6e}")
            self.log(f"  Max relative error: {valid_rel.max().item():.6e}")
            self.log(f"  Mean relative error: {valid_rel.mean().item():.6e}")
            self.log(f"  NaN count: {torch.isnan(output).sum().item()}")

            # FP8 has ~7% max relative error for values in range
            success = valid_rel.max().item() < 0.15  # 15% tolerance
            self.check_result("Roundtrip error analysis", success)
        else:
            self.check_result("Roundtrip error analysis", False, "No valid values")

    # =========================================================================
    # Test 10: FP8 Format Exhaustive Small Range
    # =========================================================================
    def test_exhaustive_small_range(self):
        self.log("\n=== Test 10: Exhaustive Small Range Test ===")

        # Test all FP16 values in a small range (exhaustive)
        # This will catch any systematic issues
        start, end = -128, 128  # Test 256 integer values
        test_values = torch.arange(start, end, device=self.device, dtype=torch.float16)

        scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
        x_fp8 = torch.zeros(len(test_values), device=self.device, dtype=torch.float8_e4m3fnuz)

        _C.quantize_per_tensor_fp8(
            test_values,
            scale,
            x_fp8,
            self.DTYPE_F16,
            self.DTYPE_F8_E4M3,
            len(test_values),
            self.stream_ptr,
        )
        torch.cuda.synchronize()

        # Check for NaN
        fp8_bytes = x_fp8.view(torch.uint8)
        nan_mask = self._detect_fp8_nan(fp8_bytes)
        nan_indices = nan_mask.nonzero(as_tuple=True)[0]

        if len(nan_indices) > 0:
            self.log(f"Found {len(nan_indices)} NaN values:")
            for idx in nan_indices[:10]:
                i = idx.item()
                self.log(f"  Input {test_values[i].item()} -> FP8 0x{fp8_bytes[i].item():02x}")

        # Verify monotonicity (quantization should preserve order)
        output_f16 = torch.zeros(len(test_values), device=self.device, dtype=torch.float16)
        _C.dequantize_per_tensor_fp8(
            x_fp8,
            scale,
            output_f16,
            self.DTYPE_F8_E4M3,
            self.DTYPE_F16,
            len(test_values),
            self.stream_ptr,
        )
        torch.cuda.synchronize()

        # Check for NaN in output
        output_nan = torch.isnan(output_f16).sum().item()

        success = (len(nan_indices) == 0) and (output_nan == 0)
        self.check_result(
            f"Exhaustive range [{start}, {end})",
            success,
            f"Quant NaN: {len(nan_indices)}, Dequant NaN: {output_nan}" if not success else "",
        )

    # =========================================================================
    # Test 11: Mixed Precision Paths
    # =========================================================================
    def test_mixed_precision(self):
        self.log("\n=== Test 11: Mixed Precision Input/Output ===")

        size = 256
        x_f32 = torch.randn(size, device=self.device, dtype=torch.float32) * 10
        x_f16 = x_f32.to(torch.float16)
        x_bf16 = x_f32.to(torch.bfloat16)
        scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)

        # Test FP32 input -> FP8
        fp8_out = torch.zeros(size, device=self.device, dtype=torch.float8_e4m3fnuz)
        _C.quantize_per_tensor_fp8(
            x_f32, scale, fp8_out, self.DTYPE_F32, self.DTYPE_F8_E4M3, size, self.stream_ptr
        )
        torch.cuda.synchronize()
        nan_count_f32 = self._detect_fp8_nan(fp8_out.view(torch.uint8)).sum().item()

        # Test FP16 input -> FP8
        _C.quantize_per_tensor_fp8(
            x_f16, scale, fp8_out, self.DTYPE_F16, self.DTYPE_F8_E4M3, size, self.stream_ptr
        )
        torch.cuda.synchronize()
        nan_count_f16 = self._detect_fp8_nan(fp8_out.view(torch.uint8)).sum().item()

        # Test BF16 input -> FP8
        _C.quantize_per_tensor_fp8(
            x_bf16, scale, fp8_out, self.DTYPE_BF16, self.DTYPE_F8_E4M3, size, self.stream_ptr
        )
        torch.cuda.synchronize()
        nan_count_bf16 = self._detect_fp8_nan(fp8_out.view(torch.uint8)).sum().item()

        success = (nan_count_f32 == 0) and (nan_count_f16 == 0) and (nan_count_bf16 == 0)
        self.check_result(
            "Mixed precision inputs",
            success,
            f"NaN: FP32={nan_count_f32}, FP16={nan_count_f16}, BF16={nan_count_bf16}"
            if not success
            else "",
        )

        # Test FP8 -> different output precisions
        fp8_input = x_f16.to(torch.float8_e4m3fnuz)

        for dtype, dtype_code, name in [
            (torch.float32, self.DTYPE_F32, "FP32"),
            (torch.float16, self.DTYPE_F16, "FP16"),
            (torch.bfloat16, self.DTYPE_BF16, "BF16"),
        ]:
            output = torch.zeros(size, device=self.device, dtype=dtype)
            _C.dequantize_per_tensor_fp8(
                fp8_input, scale, output, self.DTYPE_F8_E4M3, dtype_code, size, self.stream_ptr
            )
            torch.cuda.synchronize()

            nan_count = torch.isnan(output.float()).sum().item()
            success = nan_count == 0
            self.check_result(
                f"FP8 -> {name} output", success, f"NaN count: {nan_count}" if not success else ""
            )

    # =========================================================================
    # Utility Functions
    # =========================================================================
    def _detect_fp8_nan(self, fp8_bytes: torch.Tensor) -> torch.Tensor:
        """Detect NaN values in FP8 E4M3 format."""
        # E4M3: exponent bits are [6:3], mantissa bits are [2:0]
        exp = (fp8_bytes >> 3) & 0x0F
        mant = fp8_bytes & 0x07
        return (exp == 0x0F) & (mant != 0x00)

    def _is_fp8_nan_single(self, byte_val: int) -> bool:
        """Check if a single FP8 byte is NaN."""
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07
        return (exp == 0x0F) and (mant != 0x00)

    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all(self):
        self.log("=" * 60)
        self.log("COMPREHENSIVE FP8 QUANTIZATION TEST SUITE")
        self.log("=" * 60)

        tests = [
            ("Basic Roundtrip", self.test_basic_roundtrip),
            ("Special Values", self.test_special_values),
            ("Scale Variations", self.test_scale_variations),
            ("Large Tensors", self.test_large_tensors),
            ("Alignment/Boundary", self.test_alignment),
            ("Multiple Tensors", self.test_per_tensor_multiple),
            ("Dequantize Accuracy", self.test_dequantize_accuracy),
            ("NaN Propagation", self.test_nan_propagation),
            ("Roundtrip Error Analysis", self.test_roundtrip_error_analysis),
            ("Exhaustive Small Range", self.test_exhaustive_small_range),
            ("Mixed Precision", self.test_mixed_precision),
        ]

        for name, test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                self.log(f"✗ {name} CRASHED: {e}")
                self.failed += 1
                self.errors.append((name, f"Exception: {e}"))

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)
        total = self.passed + self.failed
        self.log(f"Total: {total} | Passed: {self.passed} | Failed: {self.failed}")

        if self.errors:
            self.log("\nFailed tests:")
            for name, details in self.errors:
                self.log(f"  ✗ {name}: {details}")

        return self.failed == 0


# =============================================================================
# Quick Diagnostic Tool
# =============================================================================
class QuickDiagnostic:
    """Minimal reproduction of the NaN issue."""

    def __init__(self, device="cuda"):
        self.device = device
        self.stream_ptr = torch.cuda.current_stream().cuda_stream

    def run_diagnostic(self):
        print("=" * 60)
        print("QUICK DIAGNOSTIC: NaN Detection")
        print("=" * 60)

        # The original failing case
        print("\n1. Original test case:")
        x = torch.arange(16, device=self.device, dtype=torch.float16)
        scale = torch.tensor([1.0], device=self.device, dtype=torch.float32)
        x_fp8 = x.to(torch.float8_e4m3fnuz)
        output = torch.zeros(16, device=self.device, dtype=torch.float16)

        _C.dequantize_per_tensor_fp8(x_fp8, scale, output, 5, 1, x.numel(), self.stream_ptr)
        torch.cuda.synchronize()

        expected = x_fp8.to(torch.float16)
        print(f"Input: {x}")
        print(f"Output: {output}")
        print(f"Expected: {expected}")
        print(f"Has NaN: {torch.isnan(output).any().item()}")

        # Test quantization direction
        print("\n2. Quantize direction test:")
        test_input = torch.tensor(
            [-448.0, -449.0, 0.0, 448.0, 449.0], device=self.device, dtype=torch.float16
        )
        fp8_out = torch.zeros(5, device=self.device, dtype=torch.float8_e4m3fnuz)

        _C.quantize_per_tensor_fp8(test_input, scale, fp8_out, 1, 5, 5, self.stream_ptr)
        torch.cuda.synchronize()

        fp8_bytes = fp8_out.view(torch.uint8)
        print(f"Input values: {test_input}")
        print(
            f"FP8 bytes: 0x{fp8_bytes[0].item():02x} 0x{fp8_bytes[1].item():02x} "
            f"0x{fp8_bytes[2].item():02x} 0x{fp8_bytes[3].item():02x} 0x{fp8_bytes[4].item():02x}"
        )

        # Check for NaN pattern
        for i in range(5):
            exp = (fp8_bytes[i].item() >> 3) & 0x0F
            mant = fp8_bytes[i].item() & 0x07
            is_nan = (exp == 0x0F) and (mant != 0x00)
            print(
                f"  [{i}] val={test_input[i].item():6.1f} -> byte=0x{fp8_bytes[i].item():02x} "
                f"exp={exp} mant={mant} NaN={is_nan}"
            )

        # Systematic scan for NaN
        print("\n3. Scanning for NaN-producing values:")
        scan_range = torch.arange(-500, 501, device=self.device, dtype=torch.float16)
        fp8_scan = torch.zeros(len(scan_range), device=self.device, dtype=torch.float8_e4m3fnuz)

        _C.quantize_per_tensor_fp8(
            scan_range, scale, fp8_scan, 1, 5, len(scan_range), self.stream_ptr
        )
        torch.cuda.synchronize()

        fp8_scan_bytes = fp8_scan.view(torch.uint8)
        nan_mask = ((fp8_scan_bytes >> 3) & 0x0F == 0x0F) & ((fp8_scan_bytes & 0x07) != 0x00)
        nan_indices = nan_mask.nonzero(as_tuple=True)[0]

        if len(nan_indices) > 0:
            print(f"Found {len(nan_indices)} NaN-producing values:")
            for idx in nan_indices[:20]:
                i = idx.item()
                print(
                    f"  Input: {scan_range[i].item():.1f} -> FP8: 0x{fp8_scan_bytes[i].item():02x}"
                )
        else:
            print("No NaN found in range [-500, 500]")

        return len(nan_indices) == 0


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP8 Quantization Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick diagnostic only")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if args.quiet:
        verbose = False
    else:
        verbose = args.verbose

    if args.quick:
        diag = QuickDiagnostic()
        success = diag.run_diagnostic()
    else:
        suite = FP8TestSuite(verbose=verbose)
        success = suite.run_all()

    sys.exit(0 if success else 1)
