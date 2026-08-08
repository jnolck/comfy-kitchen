# File: tests/definitive_fp8_format_test.py
"""
DEFINITIVE AMD FNUZ FP8 FORMAT TEST
Key findings:
- AMD FNUZ E4M3 uses exponent bias = 8 (standard E4M3 uses bias = 7)
- AMD FNUZ has no negative zero (0x80 is NaN, not -0)
- Max value: 0x7f = 240.0
- This is the same format as NVidia's E4M3 but with bias shifted by 1
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


def decode_amd_e4m3fnuz(byte_val):
    """
    CORRECT decoding of AMD FNUZ E4M3 format.

    AMD FNUZ E4M3 specification:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Exponent bias = 8 (not 7 like standard E4M3)
    - No negative zero (0x80 is NaN)
    - No infinity (all exponent=1 patterns are NaN)
    """
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0x0F
    mant = byte_val & 0x07

    # AMD FNUZ specific: no negative zero
    if byte_val == 0x80:
        return float("nan")

    if exp == 0:
        # Subnormal: (-1)^sign * 2^(1-bias) * (mant/8) = 2^(-7) * mant/8
        val = (mant / 8.0) * (2**-7)
    elif exp == 0x0F:
        # NaN (AMD FNUZ has no infinity, all exponent=15 are NaN)
        return float("nan")
    else:
        # Normal: (-1)^sign * 2^(exp-bias) * (1 + mant/8)
        # bias = 8
        val = (1.0 + mant / 8.0) * (2 ** (exp - 8))

    return -val if sign else val


def encode_amd_e4m3fnuz(value):
    """
    Encode a float value to AMD FNUZ E4M3 format.
    Uses round-to-nearest-even.
    """
    if value == 0.0:
        return 0x00
    if np.isnan(value):
        return 0x7F  # Canonical NaN
    if np.isinf(value):
        return 0x7F  # Clamp to NaN

    sign = 0
    if value < 0:
        sign = 1
        value = -value

    # Clamp to max
    if value >= 240.0:
        return 0x7F if sign == 0 else 0xFF

    # Find exponent
    exp = int(np.floor(np.log2(value))) + 8  # bias = 8
    if exp <= 0:
        # Subnormal
        mant = int(round(value / (2**-7) * 8))
        if mant >= 8:
            exp = 1
            mant = 0
            return (sign << 7) | (exp << 3) | mant
        mant = min(mant, 7)
        return (sign << 7) | mant

    if exp >= 15:
        return 0x7F if sign == 0 else 0xFF

    # Normal
    mantissa_val = value / (2 ** (exp - 8)) - 1.0
    mant = int(round(mantissa_val * 8))

    if mant == 8:
        exp += 1
        mant = 0
        if exp >= 15:
            return 0x7F if sign == 0 else 0xFF

    return (sign << 7) | (exp << 3) | mant


def verify_decoding():
    """Verify our decoding matches PyTorch ROCm."""
    print("=" * 60)
    print("VERIFYING AMD FNUZ E4M3 DECODING (bias=8)")
    print("=" * 60)

    # Test all valid bit patterns
    all_patterns = list(range(256))
    test_patterns = all_patterns  # Can reduce to interesting ones

    mismatches = 0
    for pattern in test_patterns:
        if pattern == 0x80:
            continue  # Skip negative zero (known issue)

        t = torch.tensor([pattern], dtype=torch.uint8).view(torch.float8_e4m3fnuz)
        pytorch_val = t.to(torch.float16).item()
        decoded_val = decode_amd_e4m3fnuz(pattern)

        if np.isnan(pytorch_val) and np.isnan(decoded_val):
            continue
        if abs(pytorch_val - decoded_val) > 0.01:
            mismatches += 1
            if mismatches <= 10:
                print(f"  0x{pattern:02x}: PyTorch={pytorch_val:.6f}, Decoded={decoded_val:.6f}")

    if mismatches == 0:
        print("✓ All patterns match (bias=8 decoding is correct)")
    else:
        print(f"✗ {mismatches} mismatches found")

    # Show key values
    print("\nKey values with bias=8:")
    test_vals = [0.0, 0.5, 1.0, 2.0, 4.0, 128.0, 240.0]
    print(f"{'Value':>8} {'Byte':>6} {'Decoded':>10} {'PyTorch':>10}")
    print("-" * 40)
    for val in test_vals:
        t = torch.tensor([val], dtype=torch.float16).to(torch.float8_e4m3fnuz)
        byte_val = t.view(torch.uint8).item()
        decoded = decode_amd_e4m3fnuz(byte_val)
        pytorch_val = t.to(torch.float16).item()
        print(f"{val:8.1f} 0x{byte_val:02x}   {decoded:10.4f} {pytorch_val:10.4f}")


def verify_encoding():
    """Verify our encoding matches PyTorch ROCm."""
    print("\n" + "=" * 60)
    print("VERIFYING AMD FNUZ E4M3 ENCODING")
    print("=" * 60)

    test_values = [
        -240.0,
        -128.0,
        -64.0,
        -32.0,
        -16.0,
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        240.0,
    ]

    all_match = True
    for val in test_values:
        t = torch.tensor([val], dtype=torch.float16).to(torch.float8_e4m3fnuz)
        expected_byte = t.view(torch.uint8).item()
        encoded_byte = encode_amd_e4m3fnuz(val)

        match = encoded_byte == expected_byte
        if not match:
            all_match = False
            print(f"  {val:8.1f}: encoded=0x{encoded_byte:02x}, expected=0x{expected_byte:02x} ✗")

    if all_match:
        print("✓ All test values encode correctly")
    else:
        print("✗ Some encoding mismatches")


def test_kernel_with_correct_format():
    """Test the kernel knowing the correct format."""
    print("\n" + "=" * 60)
    print("TESTING KERNEL WITH AMD FNUZ FORMAT")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Values within AMD FNUZ range (max=240)
    safe_values = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 240.0],
        device=device,
        dtype=torch.float16,
    )

    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Quantize test
    output_uint8 = torch.zeros(safe_values.shape, device=device, dtype=torch.uint8)
    _C.quantize_per_tensor_fp8(
        safe_values,
        scale,
        output_uint8,
        1,
        5,  # FP16 -> FP8
        safe_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Compare with PyTorch
    expected_fp8 = safe_values.to(torch.float8_e4m3fnuz)
    expected_bytes = expected_fp8.view(torch.uint8)

    matches = output_uint8 == expected_bytes
    print(f"Quantize matches: {matches.sum().item()}/{len(safe_values)}")

    if not matches.all():
        mismatches = [i for i, m in enumerate(matches) if not m]
        for i in mismatches:
            val = safe_values[i].item()
            got_byte = output_uint8[i].item()
            exp_byte = expected_bytes[i].item()
            got_decoded = decode_amd_e4m3fnuz(got_byte)
            exp_decoded = decode_amd_e4m3fnuz(exp_byte)
            print(
                f"  [{i}] {val:.1f}: kernel=0x{got_byte:02x}({got_decoded:.4f}) "
                f"pytorch=0x{exp_byte:02x}({exp_decoded:.4f})"
            )

    # Dequantize test
    fp8_input = safe_values.to(torch.float8_e4m3fnuz)
    output_f16 = torch.zeros(safe_values.shape, device=device, dtype=torch.float16)

    _C.dequantize_per_tensor_fp8(
        fp8_input.view(torch.uint8),
        scale,
        output_f16,
        5,
        1,  # FP8 -> FP16
        safe_values.numel(),
        stream_ptr,
    )
    torch.cuda.synchronize()

    nan_count = torch.isnan(output_f16).sum().item()
    expected_f16 = fp8_input.to(torch.float16)
    max_diff = (
        (output_f16 - expected_f16).abs()[~torch.isnan(output_f16)].max().item()
        if nan_count < len(safe_values)
        else float("nan")
    )

    print(f"Dequantize NaN count: {nan_count}")
    print(f"Dequantize max diff: {max_diff:.6f}")

    return matches.all().item() and nan_count == 0


def diagnose_kernel_format():
    """
    Determine exactly what format the kernel is producing.
    The kernel might be using standard E4M3 (bias=7) or AMD FNUZ (bias=8).
    """
    print("\n" + "=" * 60)
    print("DIAGNOSING KERNEL OUTPUT FORMAT")
    print("=" * 60)

    device = "cuda"
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Test with known values to determine bias
    test_input = torch.tensor([1.0, 2.0, 4.0, 0.5], device=device, dtype=torch.float16)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    output_uint8 = torch.zeros(test_input.shape, device=device, dtype=torch.uint8)
    _C.quantize_per_tensor_fp8(
        test_input, scale, output_uint8, 1, 5, test_input.numel(), stream_ptr
    )
    torch.cuda.synchronize()

    print(
        f"{'Input':>8} {'Kernel Byte':>14} {'If bias=7':>12} {'If bias=8':>12} {'PyTorch Byte':>14}"
    )
    print("-" * 65)

    for i in range(len(test_input)):
        val = test_input[i].item()
        k_byte = output_uint8[i].item()

        # Decode with bias=7 (standard E4M3)
        exp_b7 = (k_byte >> 3) & 0x0F
        mant = k_byte & 0x07
        if exp_b7 > 0:
            decoded_b7 = (1.0 + mant / 8.0) * (2 ** (exp_b7 - 7))
        else:
            decoded_b7 = (mant / 8.0) * (2**-6)

        # Decode with bias=8 (AMD FNUZ)
        exp_b8 = (k_byte >> 3) & 0x0F
        if exp_b8 > 0:
            decoded_b8 = (1.0 + mant / 8.0) * (2 ** (exp_b8 - 8))
        else:
            decoded_b8 = (mant / 8.0) * (2**-7)

        # PyTorch reference
        pt_byte = test_input[i : i + 1].to(torch.float8_e4m3fnuz).view(torch.uint8).item()

        match_b7 = "✓" if abs(decoded_b7 - val) < 0.01 else ""
        match_b8 = "✓" if abs(decoded_b8 - val) < 0.01 else ""

        print(
            f"{val:8.1f} 0x{k_byte:02x}           {decoded_b7:8.1f} {match_b7}  {decoded_b8:8.1f} {match_b8}  0x{pt_byte:02x}"
        )

    print("\nAnalysis:")
    k_byte_1 = output_uint8[test_input == 1.0].item()
    if k_byte_1 == 0x40:
        print("  1.0 -> 0x40: This matches BOTH bias=7 (exp=8, 8-7=1) AND bias=8 (exp=8, 8-8=0)")
        print("     Need to check 0.5 to distinguish:")
        k_byte_05 = output_uint8[test_input == 0.5].item()
        exp_05 = (k_byte_05 >> 3) & 0x0F
        if exp_05 == 7:
            print(f"    0.5 -> 0x{k_byte_05:02x}: exp={exp_05}")
            print("    exp=7: bias=8 gives 7-8=-1, 2^-1=0.5 → AMD FNUZ (bias=8)")
        elif exp_05 == 6:
            print(f"    0.5 -> 0x{k_byte_05:02x}: exp={exp_05}")
            print("    exp=6: bias=7 gives 6-7=-1, 2^-1=0.5 → Standard E4M3 (bias=7)")


if __name__ == "__main__":
    verify_decoding()
    verify_encoding()
    diagnose_kernel_format()
    test_kernel_with_correct_format()
