# File: tests/test_nvfp4_accuracy.py
"""Test NVFP4 dequant against a pure-Python reference decoder."""

import torch
import sys
import numpy as np

sys.path.insert(0, "/var/home/jnolck/Documents/src/ai/comfy-kitchen/comfy_kitchen/backends/hip")
import importlib.util

spec = importlib.util.spec_from_file_location("_C", "_C.abi3.so")
_C = importlib.util.module_from_spec(spec)
sys.modules["_C"] = _C
spec.loader.exec_module(_C)

device = "cuda"


def decode_nvfp4_nibble(nibble: int) -> float:
    """NVFP4 E2M1 format: 1 sign, 2 exp, 1 mant, bias=1."""
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1

    if exp == 0:
        val = 0.0 if mant == 0 else 0.5
    elif exp == 1:
        val = 1.0 + mant * 0.5
    elif exp == 2:
        val = 2.0 + mant * 1.0
    else:
        val = 4.0 + mant * 2.0

    return -val if sign else val


def dequant_nvfp4_reference(qx_bytes, per_tensor_scale, block_scales, hi_first=True):
    """Pure Python NVFP4 dequant for validation."""
    if isinstance(block_scales, torch.Tensor):
        block_scales = block_scales.cpu().float().numpy()
    num_rows, num_cols_packed = qx_bytes.shape
    num_cols = num_cols_packed * 2
    output = np.zeros((num_rows, num_cols), dtype=np.float32)

    block_size = 16

    for r in range(num_rows):
        for c in range(num_cols):
            byte_idx = c // 2
            nibble_idx = c % 2
            byte_val = qx_bytes[r, byte_idx]
            if hi_first:
                nibble = (byte_val >> 4) if nibble_idx == 0 else (byte_val & 0x0F)
            else:
                nibble = (byte_val & 0x0F) if nibble_idx == 0 else (byte_val >> 4)

            val = decode_nvfp4_nibble(nibble)
            block_idx = c // block_size
            block_scale = float(block_scales[r, block_idx])
            per_scale = float(per_tensor_scale)
            output[r, c] = val * block_scale * per_scale

    return output


print("=" * 60)
print("NVFP4 DEQUANT ACCURACY TEST")
print("=" * 60)

num_rows, num_cols = 16, 64

# Pack deterministic nibbles
qx_np = np.zeros((num_rows, num_cols // 2), dtype=np.uint8)
for r in range(num_rows):
    for i in range(num_cols // 2):
        qx_np[r, i] = ((i * 2) % 16) << 4 | ((i * 2 + 1) % 16)

qx = torch.from_numpy(qx_np).to(device)
per_tensor_scale = torch.tensor([2.0], device=device, dtype=torch.float32)

# Pad block scales to swizzled layout dimensions
M_padded = ((num_rows + 127) // 128) * 128
N_blocks = num_cols // 16
N_padded = ((N_blocks + 3) // 4) * 4

block_scales_f32 = torch.zeros(M_padded, N_padded, device=device, dtype=torch.float32)
block_scales_f32[:] = 3.0  # Fill entire padded tensor
block_scales_fp8 = block_scales_f32.to(torch.float8_e4m3fn)

# Direct C++ call (bypasses Python wrapper)
stream = torch.cuda.current_stream().cuda_stream
output_hip = torch.zeros(num_rows, num_cols, device=device, dtype=torch.float32)
_C.dequantize_nvfp4(
    qx,
    per_tensor_scale,
    block_scales_fp8.view(torch.uint8),
    output_hip,
    0,  # float32
    True,  # hi_first
    stream,
)
torch.cuda.synchronize()

# Reference (uses unpadded block scales)
block_scales_ref = torch.ones(num_rows, N_blocks, dtype=torch.float32) * 3.0
output_ref = dequant_nvfp4_reference(qx_np, 2.0, block_scales_ref)
output_hip_np = output_hip.cpu().float().numpy()

print(f"per_tensor_scale: {per_tensor_scale.item()}")
diff = np.abs(output_hip_np - output_ref)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"Exact matches: {(diff < 0.001).sum()}/{output_ref.size}")

if diff.max() > 0.01:
    print("\nFirst 10 mismatches:")
    count = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if diff[r, c] > 0.01:
                print(
                    f"  [{r},{c}] HIP={output_hip_np[r, c]:.4f} REF={output_ref[r, c]:.4f} diff={diff[r, c]:.4f}"
                )
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break

print(f"\nSample values (row 0):")
print(f"  HIP:     {output_hip_np[0, :16]}")
print(f"  REF:     {output_ref[0, :16]}")
print(f"\nSample values (row 4):")
print(f"  HIP:     {output_hip_np[4, :16]}")
print(f"  REF:     {output_ref[4, :16]}")
