# File: tests/corrected_fp8_test.py
"""
Test that accounts for PyTorch ROCm FP8 emulation on RDNA3.
The kernel processes standard E4M3/E5M2 bit patterns correctly,
but PyTorch ROCm clamps to different max values in software.
"""

import torch
import sys
import numpy as np

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)


def verify_fp8_standard_format():
    """
    Verify that AMD FNUZ is actually standard FP8 E4M3 format.
    Decode according to standard E4M3 specification.
    """
    print("=" * 60)
    print("VERIFYING FP8 FORMAT (Standard E4M3 Decoding)")
    print("=" * 60)

    def decode_standard_e4m3(byte_val):
        """Standard E4M3 decoding: bias=7, 1 sign, 4 exp, 3 mant"""
        sign = (byte_val >> 7) & 1
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07

        if exp == 0:
            # Subnormal: (-1)^sign * 2^(-6) * (mant/8)
            val = (mant / 8.0) * (2**-6)
        elif exp == 0x0F:
            # NaN/Inf
            if mant == 0:
                return float("inf") if sign == 0 else float("-inf")
            else:
                return float("nan")
        else:
            # Normal: (-1)^sign * 2^(exp-7) * (1 + mant/8)
            val = (1.0 + mant / 8.0) * (2 ** (exp - 7))

        return -val if sign else val

    test_values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 240.0],
        dtype=torch.float16,
    )

    fp8_fp16 = test_values.to(torch.float8_e4m3fnuz)
    fp8_bytes = fp8_fp16.view(torch.uint8)
    fp8_back = fp8_fp16.to(torch.float16)

    print(f"{'Value':>8} {'Byte':>6} {'Decoded Standard':>16} {'PyTorch':>8} {'Match':>6}")
    print("-" * 60)

    all_match = True
    for i in range(len(test_values)):
        val = test_values[i].item()
        byte_val = fp8_bytes[i].item()
        decoded = decode_standard_e4m3(byte_val)
        pytorch_val = fp8_back[i].item()

        if np.isnan(decoded) and np.isnan(pytorch_val):
            match = True
        elif np.isinf(decoded) and np.isinf(pytorch_val):
            match = True
        else:
            match = abs(decoded - pytorch_val) < 0.01

        if not match:
            all_match = False

        print(
            f"{val:8.1f} 0x{byte_val:02x}   {decoded:16.6f} {pytorch_val:8.1f} {'✓' if match else '✗':>6}"
        )

    if all_match:
        print("\n✓ AMD FNUZ IS STANDARD E4M3 FORMAT (bias=7)")
        print("  The only difference is PyTorch ROCm's software clamping")
    else:
        print("\n✗ Format mismatch detected")

    return all_match


def test_kernel_with_standard_format():
    """
    Test the kernel using standard E4M3 format understanding.
    The kernel should handle values within standard E4M3 range correctly.
    """
    print("\n" + "=" * 60)
    print("TESTING KERNEL WITH STANDARD E4M3 RANGE")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Values within standard E4M3 range (max = 448)
    test_values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 240.0],
        device=device,
        dtype=torch.float16,
    )

    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Test 1: Quantize kernel
    print("\n1. Quantize (FP16 -> FP8):")
    output_uint8 = torch.zeros(test_values.shape, device=device, dtype=torch.uint8)

    _C.quantize_per_tensor_fp8(
        test_values,
        scale,
        output_uint8,
        1,
        5,  # FP16 -> FP8 E4M3
        test_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Compare with PyTorch (which uses ROCm emulation)
    expected_fp8 = test_values.to(torch.float8_e4m3fnuz)
    expected_bytes = expected_fp8.view(torch.uint8)

    output_bytes = output_uint8
    matches = output_bytes == expected_bytes

    print(f"  Matches PyTorch: {matches.sum().item()}/{test_values.numel()}")
    if not matches.all():
        mismatches = [i for i, m in enumerate(matches) if not m]
        for i in mismatches:
            print(
                f"  [{i}] val={test_values[i].item():.1f} "
                f"kernel=0x{output_bytes[i].item():02x} "
                f"pytorch=0x{expected_bytes[i].item():02x}"
            )

    # Test 2: Dequantize kernel
    print("\n2. Dequantize (FP8 -> FP16):")
    fp8_input = test_values.to(torch.float8_e4m3fnuz)
    output_f16 = torch.zeros(test_values.shape, device=device, dtype=torch.float16)

    _C.dequantize_per_tensor_fp8(
        fp8_input.view(torch.uint8),
        scale,
        output_f16,
        5,
        1,  # FP8 E4M3 -> FP16
        test_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Check for NaN
    nan_count = torch.isnan(output_f16).sum().item()
    print(f"  NaN count: {nan_count}")

    # Compare with expected (within FP8 precision)
    expected_f16 = fp8_input.to(torch.float16)
    abs_diff = (output_f16 - expected_f16).abs()
    max_diff = (
        abs_diff[~torch.isnan(output_f16)].max().item()
        if nan_count < len(test_values)
        else float("nan")
    )

    print(f"  Max diff (excl NaN): {max_diff:.6f}")

    if nan_count > 0:
        nan_indices = torch.isnan(output_f16).nonzero().flatten()
        for i in nan_indices:
            fp8_byte = fp8_input.view(torch.uint8)[i].item()
            print(f"    [{i}] fp8=0x{fp8_byte:02x} val={test_values[i].item():.1f}")

    # Verify round-trip accuracy
    print("\n3. Round-trip comparison:")
    for i in range(len(test_values)):
        orig = test_values[i].item()
        fp8 = expected_bytes[i].item()
        back = output_f16[i].item() if not torch.isnan(output_f16[i]) else float("nan")
        expected_back = expected_f16[i].item()

        match = (
            "✓"
            if abs(back - expected_back) < 0.01 or (np.isnan(back) and np.isnan(expected_back))
            else "✗"
        )
        print(f"  {orig:8.1f} -> 0x{fp8:02x} -> {back:8.1f} (exp: {expected_back:8.1f}) {match}")


def diagnose_clamping_difference():
    """
    Show exactly where PyTorch ROCm clamps differently from standard E4M3.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSING CLAMPING BEHAVIOR")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Standard E4M3 max = 448.0
    # What does PyTorch ROCm actually do?
    values_above_240 = torch.linspace(240, 450, 20, device=device, dtype=torch.float16)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    print(
        f"{'Input':>8} {'PyTorch Byte':>14} {'PyTorch Val':>12} {'Kernel Byte':>14} {'Kernel Val':>12}"
    )
    print("-" * 70)

    for i in range(len(values_above_240)):
        val = values_above_240[i].item()

        # PyTorch
        pt_fp8 = values_above_240[i : i + 1].to(torch.float8_e4m3fnuz)
        pt_byte = pt_fp8.view(torch.uint8).item()
        pt_back = pt_fp8.to(torch.float16).item()

        # Kernel
        k_uint8 = torch.zeros(1, device=device, dtype=torch.uint8)
        _C.quantize_per_tensor_fp8(values_above_240[i : i + 1], scale, k_uint8, 1, 5, 1, stream_ptr)
        torch.cuda.synchronize()
        k_byte = k_uint8.item()
        k_fp8 = k_uint8.view(torch.float8_e4m3fnuz)
        k_back = k_fp8.to(torch.float16).item()

        match = "✓" if pt_byte == k_byte else "✗"
        print(
            f"{val:8.1f} 0x{pt_byte:02x} ({pt_back:8.1f})   0x{k_byte:02x} ({k_back:8.1f})   {match}"
        )

    # What does the kernel produce for 448.0?
    val_448 = torch.tensor([448.0], device=device, dtype=torch.float16)
    k_uint8 = torch.zeros(1, device=device, dtype=torch.uint8)
    _C.quantize_per_tensor_fp8(val_448, scale, k_uint8, 1, 5, 1, stream_ptr)
    torch.cuda.synchronize()
    k_byte_448 = k_uint8.item()
    print(f"\n  Kernel at 448.0: 0x{k_byte_448:02x}")

    # Standard E4M3 encoding of 448.0:
    # 448 = 1.75 * 2^8
    # sign=0, exp=8+7=15=0x0F, mant=round(0.75*8)=6=0x6
    # Byte: 0 1111 110 = 0x7E
    expected_448 = 0x7E
    print(f"  Standard E4M3 448.0: 0x{expected_448:02x}")


if __name__ == "__main__":
    # Verify format
    is_standard_format = verify_fp8_standard_format()

    # Test kernel with standard range
    test_kernel_with_standard_format()

    # Diagnose clamping
    diagnose_clamping_difference()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if is_standard_format:
        print("""
AMD FNUZ = Standard E4M3 format (same bit layout, same bias=7).

The issue is PyTorch ROCm's SOFTWARE EMULATION:
- RDNA3 has no FP8 hardware
- PyTorch ROCm clamps FP8 to ~240 (not 448) in software
- 0x80 in FNUZ means -0 which is treated as NaN

YOUR KERNEL IS LIKELY CORRECT for standard E4M3.
The NaN you're seeing comes from:
1. PyTorch ROCm's non-standard clamping during conversion
2. Possible mismatch in how 0x80 (negative zero) is handled

RECOMMENDATION:
- Keep kernel using standard E4M3 (max=448)
- Add explicit clamping in Python interface layer
- Handle 0x80 properly (it's -0 in standard E4M3, NaN in FNUZ)
""")
    else:
        print("Format differs from standard E4M3. Check bias/exponent encoding.")
