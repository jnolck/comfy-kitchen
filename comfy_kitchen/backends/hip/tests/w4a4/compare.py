import torch
import comfy_kitchen as ck
from comfy_kitchen.backends.eager.convrot_w4a4 import (
    _build_hadamard,
    _rotate_activation,
    _rotate_weight,
    quantize_signed_int4_rowwise,
    int4_linear,
    _unpack_int4_row_major,
)
from comfy_kitchen.backends.hip import prepare_int4_weight_for_int8_linear

torch.manual_seed(42)
M, K, N = 16, 256, 64
device = "cuda"
dtype = torch.bfloat16
group_size = 256

print(f"=== ConvRot W4A4 Fallback Debug: M={M}, K={K}, N={N} ===\n")

# Create inputs
x = torch.randn(M, K, dtype=dtype, device=device)
weight = torch.randn(N, K, dtype=dtype, device=device)
h = _build_hadamard(group_size, device=device, dtype=dtype)

# ---- Step 1: Compare rotation ----
x_rot_eager = _rotate_activation(x, h, group_size)
x_rot_hip = _rotate_activation(x, h, group_size)  # same math, should match
print(f"1. Rotation match: {torch.allclose(x_rot_eager.float(), x_rot_hip.float(), atol=0.01)}")

# ---- Step 2: Compare quantization ----
qact_eager, x_scale_eager = quantize_signed_int4_rowwise(x_rot_eager)

from comfy_kitchen.backends.hip import quantize_int4_rowwise, convrot_w4a4_linear

qact_hip, x_scale_hip = quantize_int4_rowwise(x_rot_eager.contiguous())

print(f"2. Quantization:")
print(f"   qact match: {torch.equal(qact_eager, qact_hip)}")
print(f"   qact max diff: {(qact_eager.int() - qact_hip.int()).abs().max().item()}")
print(f"   x_scale match: {torch.allclose(x_scale_eager, x_scale_hip, atol=0.01)}")
print(f"   x_scale max diff: {(x_scale_eager - x_scale_hip).abs().max().item():.6f}")

# ---- Step 3: Compare unpacked activations ----
act_eager_unpacked = _unpack_int4_row_major(qact_eager).float()
act_hip_unpacked = _unpack_int4_row_major(qact_hip).float()
print(f"3. Unpacked activations:")
print(f"   max diff: {(act_eager_unpacked - act_hip_unpacked).abs().max().item()}")
print(f"   mean diff: {(act_eager_unpacked - act_hip_unpacked).abs().mean().item():.4f}")

# ---- Step 4: Compare weight quantization ----
w_rot = _rotate_weight(weight, h, group_size)
qw_eager, w_scale_eager = quantize_signed_int4_rowwise(w_rot)

qw_hip, w_scale_hip = quantize_int4_rowwise(w_rot.contiguous())

print(f"4. Weight quantization:")
print(f"   qw match: {torch.equal(qw_eager, qw_hip)}")
print(f"   qw max diff: {(qw_eager.int() - qw_hip.int()).abs().max().item()}")
print(f"   w_scale match: {torch.allclose(w_scale_eager, w_scale_hip, atol=0.01)}")

# ---- Step 5: Compare full pipeline with SAME quantized inputs ----
print(f"\n5. Full pipeline (using EAGER quantized inputs for both):")

out_eager = convrot_w4a4_linear(
    x, qw_eager, w_scale_eager, convrot_groupsize=group_size, linear_dtype="int8"
)
out_hip = convrot_w4a4_linear(
    x, qw_eager, w_scale_eager, convrot_groupsize=group_size, linear_dtype="int8"
)

diff = (out_eager.float() - out_hip.float()).abs()
print(f"   Max diff: {diff.max().item():.4f}")
print(f"   Mean diff: {diff.mean().item():.4f}")

# ---- Step 6: Full pipeline with HIP quantized inputs ----
print(f"\n6. Full pipeline (using HIP quantized inputs for both):")
out_eager2 = convrot_w4a4_linear(
    x, qw_hip, w_scale_hip, convrot_groupsize=group_size, linear_dtype="int8"
)
out_hip2 = convrot_w4a4_linear(
    x, qw_hip, w_scale_hip, convrot_groupsize=group_size, linear_dtype="int8"
)

diff2 = (out_eager2.float() - out_hip2.float()).abs()
print(f"   Max diff: {diff2.max().item():.4f}")
print(f"   Mean diff: {diff2.mean().item():.4f}")

print(f"\nFirst 8 values (eager quant):")
print(f"  eager: {out_eager[0, :8].float().tolist()}")
print(f"  hip:   {out_hip[0, :8].float().tolist()}")
print(f"\nFirst 8 values (hip quant):")
print(f"  eager: {out_eager2[0, :8].float().tolist()}")
print(f"  hip:   {out_hip2[0, :8].float().tolist()}")

# Add to quant.py
print(f"\nFirst 20 values comparison:")
print(f"HIP:   {qact_hip[0, :20].int().tolist()}")
print(f"Eager: {qact_eager[0, :20].int().tolist()}")

# Find where they differ
diff_mask = qact_hip.int() != qact_eager.int()
diff_indices = torch.where(diff_mask)
print(f"\nTotal differences: {diff_mask.sum().item()} out of {qact_hip.numel()}")
if diff_mask.sum() > 0:
    print(f"First diff at row {diff_indices[0][0].item()}, col {diff_indices[1][0].item()}")
    print(f"  HIP: {qact_hip[diff_indices[0][0], diff_indices[1][0]].item()}")
    print(f"  Eager: {qact_eager[diff_indices[0][0], diff_indices[1][0]].item()}")
