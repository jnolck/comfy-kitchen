# File: tests/test_dequant_accuracy.py
"""Test HIP dequantize vs eager backend for numerical accuracy."""

import torch
import sys

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

from comfy_kitchen.backends.hip import dequantize_per_tensor_fp8 as hip_dequant
from comfy_kitchen.backends.eager.quantization import dequantize_per_tensor_fp8 as eager_dequant

device = "cuda"
torch.manual_seed(42)

print("=" * 60)
print("HIP vs EAGER DEQUANTIZE ACCURACY")
print("=" * 60)

# Test 1: All possible FP8 values with scale=1.0
print("\n1. All 256 FP8 byte patterns (scale=1.0):")
all_bytes = torch.arange(256, device=device, dtype=torch.uint8)
fp8_tensor = all_bytes.view(torch.float8_e4m3fn)
scale = torch.tensor([1.0], device=device, dtype=torch.float32)

result_hip = hip_dequant(fp8_tensor, scale, torch.float32)
result_eager = eager_dequant(fp8_tensor, scale, torch.float32)

diff = (result_hip - result_eager).abs()
matches = diff < 0.001
print(f"  Exact matches: {matches.sum().item()}/256")
if not matches.all():
    mismatches = (~matches).nonzero(as_tuple=True)[0]
    print(f"  Mismatches: {len(mismatches)}")
    for idx in mismatches[:10]:
        i = idx.item()
        print(f"    0x{i:02x}: HIP={result_hip[i].item():.6f} EAGER={result_eager[i].item():.6f}")

# Test 2: Random values with various scales
print("\n2. Random values with various scales:")
scales = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
all_pass = True

for s in scales:
    x = torch.randn(1000, device=device, dtype=torch.float16) * 10
    x_fp8 = x.to(torch.float8_e4m3fn)
    scale_t = torch.tensor([s], device=device, dtype=torch.float32)

    result_hip = hip_dequant(x_fp8, scale_t, torch.float16)
    result_eager = eager_dequant(x_fp8, scale_t, torch.float16)

    diff = (result_hip.float() - result_eager.float()).abs()
    max_diff = diff.max().item()
    nan_hip = torch.isnan(result_hip).sum().item()
    nan_eager = torch.isnan(result_eager).sum().item()

    status = "✓" if (max_diff < 0.01 and nan_hip == 0) else "✗"
    print(
        f"  scale={s:8.4f}: max_diff={max_diff:.6f}, NaN(HIP)={nan_hip}, NaN(EAGER)={nan_eager} {status}"
    )
    if max_diff >= 0.01:
        all_pass = False

print(f"\n{'=' * 60}")
print(f"{'✓ ALL PASS' if all_pass else '✗ FAILURES DETECTED'}")
