# File: tests/test_stochastic_round.py
"""Test stochastic_round_fp8 kernel produces correct standard E4M3 bytes."""

import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

device = "cuda"
stream = torch.cuda.current_stream().cuda_stream


def decode_std_e4m3(byte_val):
    """Standard E4M3: bias=7, 1 sign, 4 exp, 3 mant."""
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0x0F
    mant = byte_val & 0x07

    if exp == 0:
        val = (mant / 8.0) * (2**-6)
    else:
        val = (1.0 + mant / 8.0) * (2 ** (exp - 7))

    return -val if sign else val


print("=" * 60)
print("STOCHASTIC ROUNDING TEST")
print("=" * 60)

# Test 1: Known values → correct bytes
print("\n1. Known value encoding (no randomness):")
test_values = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 128.0, 448.0, -448.0]

for val in test_values:
    x = torch.tensor([val], device=device, dtype=torch.float16)
    rng = torch.zeros(1, device=device, dtype=torch.uint8)  # zero random = floor

    _C.stochastic_round_fp8(
        rng,
        x,
        5,  # output_type=5 (E4M3)
        x.numel(),
        stream,
    )
    torch.cuda.synchronize()

    kernel_byte = rng[0].item()
    kernel_decoded = decode_std_e4m3(kernel_byte)

    expected = x.to(torch.float8_e4m3fn)
    expected_byte = expected.view(torch.uint8)[0].item()
    expected_decoded = decode_std_e4m3(expected_byte)

    match = "✓" if kernel_byte == expected_byte else "✗"
    print(
        f"  {val:8.1f}: kernel=0x{kernel_byte:02x}({kernel_decoded:.4f}) "
        f"expected=0x{expected_byte:02x}({expected_decoded:.4f}) {match}"
    )

# Test 2: Random values vs PyTorch
print("\n2. Random values vs PyTorch (with stochastic noise):")
torch.manual_seed(42)
x = torch.randn(1000, device=device, dtype=torch.float16) * 10
rng = torch.randint(0, 256, (1000,), device=device, dtype=torch.uint8)
rng_copy = rng.clone()

_C.stochastic_round_fp8(rng, x, 5, x.numel(), stream)
torch.cuda.synchronize()

kernel_bytes = rng
expected_fp8 = x.to(torch.float8_e4m3fn)
expected_bytes = expected_fp8.view(torch.uint8)

# With zero random, should match floor rounding
rng_zero = torch.zeros(1000, device=device, dtype=torch.uint8)
_C.stochastic_round_fp8(rng_zero, x.clone(), 5, x.numel(), stream)
torch.cuda.synchronize()

# Verify all bytes are valid E4M3 (no NaN patterns for standard)
nan_count = 0
for b in kernel_bytes.tolist():
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    # Standard E4M3: exp=15 with mant!=0 is NaN
    if exp == 15 and mant != 0:
        nan_count += 1

print(f"  NaN count: {nan_count}/1000")
print(f"  Random bytes valid: {'✓' if nan_count == 0 else '✗'}")

# Test 3: Round-trip consistency
print("\n3. Stochastic round → dequantize round-trip:")
x_test = torch.tensor([1.5, -1.5, 2.3, -2.3, 0.1, -0.1], device=device, dtype=torch.float16)
rng_test = torch.zeros(6, device=device, dtype=torch.uint8)

_C.stochastic_round_fp8(rng_test, x_test, 5, 6, stream)
torch.cuda.synchronize()

# Dequantize with our kernel
fp8_tensor = rng_test.view(torch.float8_e4m3fn)
scale = torch.tensor([1.0], device=device, dtype=torch.float32)
output = torch.zeros(6, device=device, dtype=torch.float16)
_C.dequantize_per_tensor_fp8(fp8_tensor.view(torch.uint8), scale, output, 5, 1, 6, stream)
torch.cuda.synchronize()

print(f"  Input:      {x_test.tolist()}")
print(f"  FP8 bytes:  {[f'0x{b:02x}' for b in rng_test.tolist()]}")
print(f"  Dequant:    {[f'{v:.4f}' for v in output.tolist()]}")
