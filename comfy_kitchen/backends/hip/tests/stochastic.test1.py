# File: tests/test_stochastic_vs_eager.py
"""Test HIP stochastic_round_fp8 vs eager backend."""

import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

from comfy_kitchen.backends.hip import stochastic_rounding_fp8 as hip_stochastic
from comfy_kitchen.backends.eager.quantization import stochastic_rounding_fp8 as eager_stochastic

device = "cuda"
torch.manual_seed(42)

print("=" * 60)
print("HIP vs EAGER STOCHASTIC ROUNDING")
print("=" * 60)

# Test 1: Exact match with zero random
print("\n1. Zero random (deterministic floor):")
x = torch.randn(16, device=device, dtype=torch.float16) * 10
rng_hip = torch.zeros(16, device=device, dtype=torch.uint8)
rng_eager = torch.zeros(16, device=device, dtype=torch.uint8)

result_hip = hip_stochastic(x.clone(), rng_hip, torch.float8_e4m3fn)
result_eager = eager_stochastic(x.clone(), rng_eager, torch.float8_e4m3fn)

hip_bytes = result_hip.view(torch.uint8)
eager_bytes = result_eager.view(torch.uint8)
matches = hip_bytes == eager_bytes
print(f"  Exact matches: {matches.sum().item()}/16")
if not matches.all():
    for i in range(16):
        if not matches[i]:
            print(
                f"  [{i}] {x[i].item():.4f}: HIP=0x{hip_bytes[i].item():02x} EAGER=0x{eager_bytes[i].item():02x}"
            )

# Test 2: Same random seed → same output
print("\n2. Same random values:")
torch.manual_seed(123)
x = torch.randn(1000, device=device, dtype=torch.float16) * 10
rng = torch.randint(0, 256, (1000,), device=device, dtype=torch.uint8)

result_hip = hip_stochastic(x.clone(), rng.clone(), torch.float8_e4m3fn)
result_eager = eager_stochastic(x.clone(), rng.clone(), torch.float8_e4m3fn)

hip_bytes = result_hip.view(torch.uint8)
eager_bytes = result_eager.view(torch.uint8)
matches = hip_bytes == eager_bytes
print(f"  Exact matches: {matches.sum().item()}/1000")
if matches.sum().item() < 1000:
    mismatches = (~matches).nonzero(as_tuple=True)[0]
    print(f"  First 10 mismatches:")
    for idx in mismatches[:10]:
        i = idx.item()
        print(
            f"    [{i}] val={x[i].item():.6f} HIP=0x{hip_bytes[i].item():02x} EAGER=0x{eager_bytes[i].item():02x}"
        )

# Test 3: Round-trip consistency
print("\n3. HIP stochastic → HIP dequantize round-trip:")
x_test = torch.randn(100, device=device, dtype=torch.float16) * 5
rng_test = torch.randint(0, 256, (100,), device=device, dtype=torch.uint8)

fp8_hip = hip_stochastic(x_test.clone(), rng_test.clone(), torch.float8_e4m3fn)
fp8_eager = eager_stochastic(x_test.clone(), rng_test.clone(), torch.float8_e4m3fn)

# Dequantize both with HIP dequantize
from comfy_kitchen.backends.hip import dequantize_per_tensor_fp8 as hip_dequant

scale = torch.tensor([1.0], device=device, dtype=torch.float32)

out_hip = hip_dequant(fp8_hip, scale, torch.float16)
out_eager = hip_dequant(fp8_eager, scale, torch.float16)

diff = (out_hip - out_eager).abs()
print(f"  Dequantized diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
print(f"  HIP NaN: {torch.isnan(out_hip).sum().item()}")
print(f"  EAGER NaN: {torch.isnan(out_eager).sum().item()}")

# Summary
total_matches = matches.sum().item()
print(f"\n{'=' * 60}")
print(f"Overall: {total_matches}/1000 exact byte matches")
if total_matches == 1000:
    print("✓ PERFECT MATCH")
else:
    print(f"✗ {1000 - total_matches} mismatches")
