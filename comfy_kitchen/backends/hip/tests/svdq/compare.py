"""Test HIP SVDQuant kernels against eager backend"""

import torch
import comfy_kitchen as ck

torch.manual_seed(42)

# Test dimensions matching real model usage
M, K, N = 256, 3072, 3072  # Like calls 3-5 from the logs
G = 64
R = 64

print(f"=== Testing M={M}, K={K}, N={N}, G={G}, R={R} ===\n")

device = "cuda"
dtype = torch.bfloat16

# Create test inputs
x = torch.randn(M, K, dtype=dtype, device=device)
smooth = torch.ones(K, dtype=dtype, device=device)
lora_down = torch.randn(K, R, dtype=dtype, device=device)
lora_up = torch.randn(N, R, dtype=dtype, device=device)

# Create weight (packed int4)
qweight = torch.randint(0, 256, (N, K // 2), dtype=torch.int32, device=device).to(torch.int8)
wscales = torch.randn(K // G, N, dtype=dtype, device=device)

print("--- SVDQuant quantize ---")

# Eager reference
with ck.use_backend("eager"):
    q_x_eager, ascales_eager, lora_act_eager = ck.quantize_svdquant_w4a4(
        x.clone(), smooth.clone(), lora_down.clone(), act_unsigned=False
    )
print(
    f"  eager: q_x={q_x_eager.shape}, ascales={ascales_eager.shape}, lora_act={lora_act_eager.shape}"
)

# HIP backend
with ck.use_backend("hip"):
    q_x_hip, ascales_hip, lora_act_hip = ck.quantize_svdquant_w4a4(
        x.clone(), smooth.clone(), lora_down.clone(), act_unsigned=False
    )
print(f"  hip:   q_x={q_x_hip.shape}, ascales={ascales_hip.shape}, lora_act={lora_act_hip.shape}")

# Compare quantize outputs
print("\nQuantize comparison:")
print(f"  q_x match:     {torch.equal(q_x_eager, q_x_hip)}")
print(f"  ascales close: {torch.allclose(ascales_eager.float(), ascales_hip.float(), atol=0.01)}")
print(f"  lora_act close:{torch.allclose(lora_act_eager.float(), lora_act_hip.float(), atol=0.01)}")

if not torch.equal(q_x_eager, q_x_hip):
    diff = (q_x_eager.int() - q_x_hip.int()).abs()
    print(f"  q_x max diff:  {diff.max().item()}")
    print(f"  q_x diff idxs: {torch.where(diff > 0)[0][:10].tolist()}")

print("\n--- SVDQuant scaled_mm ---")

# Eager reference
with ck.use_backend("eager"):
    out_eager = ck.scaled_mm_svdquant_w4a4(
        q_x_eager, qweight, ascales_eager, wscales, lora_act_eager, lora_up
    )
print(
    f"  eager out: {out_eager.shape}, range=[{out_eager.min().item():.4f}, {out_eager.max().item():.4f}]"
)

# HIP backend
with ck.use_backend("hip"):
    out_hip = ck.scaled_mm_svdquant_w4a4(
        q_x_hip, qweight, ascales_hip, wscales, lora_act_hip, lora_up
    )
print(
    f"  hip out:   {out_hip.shape}, range=[{out_hip.min().item():.4f}, {out_hip.max().item():.4f}]"
)
# Find where q_x differs
diff_mask = q_x_eager.int() != q_x_hip.int()
diff_indices = torch.where(diff_mask)
print(f"\nDetailed q_x comparison:")
for i in range(min(5, len(diff_indices[0]))):
    r, c = diff_indices[0][i].item(), diff_indices[1][i].item()
    eager_byte = q_x_eager[r, c].item()
    hip_byte = q_x_hip[r, c].item()
    # Unpack the bytes
    eager_lo = eager_byte & 0xF
    eager_hi = (eager_byte >> 4) & 0xF
    hip_lo = hip_byte & 0xF
    hip_hi = (hip_byte >> 4) & 0xF
    # Sign extend
    if eager_lo >= 8:
        eager_lo -= 16
    if eager_hi >= 8:
        eager_hi -= 16
    if hip_lo >= 8:
        hip_lo -= 16
    if hip_hi >= 8:
        hip_hi -= 16
    print(f"  [{r},{c}]: eager=({eager_lo},{eager_hi}) hip=({hip_lo},{hip_hi})")
# Compare
diff = (out_eager.float() - out_hip.float()).abs()
print(f"\nOutput comparison:")
print(f"  Max diff:  {diff.max().item():.6f}")
print(f"  Mean diff: {diff.mean().item():.6f}")
print(f"  All close (atol=1.0): {torch.allclose(out_eager.float(), out_hip.float(), atol=1.0)}")
print(f"  All close (atol=0.1): {torch.allclose(out_eager.float(), out_hip.float(), atol=0.1)}")

# Print first few values from both
print(f"\nFirst 8 values of row 0:")
print(f"  eager: {out_eager[0, :8].float().tolist()}")
print(f"  hip:   {out_hip[0, :8].float().tolist()}")

# Find the positions with huge values
huge_mask = out_hip.float().abs() > 100000
huge_indices = torch.where(huge_mask)
print(f"\nHuge values (>100k): {huge_mask.sum().item()} positions")
for i in range(min(5, len(huge_indices[0]))):
    r, c = huge_indices[0][i].item(), huge_indices[1][i].item()
    print(f"  [{r},{c}]: hip={out_hip[r, c].item():.1f}, eager={out_eager[r, c].item():.1f}")

# In compare.py, after quantize:
# Use eager's quantized outputs for BOTH scaled_mm calls
with ck.use_backend("eager"):
    out_eager_same_input = ck.scaled_mm_svdquant_w4a4(
        q_x_eager, qweight, ascales_eager, wscales, lora_act_eager, lora_up
    )

with ck.use_backend("hip"):
    out_hip_same_input = ck.scaled_mm_svdquant_w4a4(
        q_x_eager, qweight, ascales_eager, wscales, lora_act_eager, lora_up
    )

diff_same = (out_eager_same_input.float() - out_hip_same_input.float()).abs()
print(f"\nWith same inputs (eager quantized):")
print(f"  Max diff:  {diff_same.max().item():.6f}")
print(f"  Mean diff: {diff_same.mean().item():.6f}")

diff_signed = out_eager.float() - out_hip.float()
print(f"\nSystematic bias check:")
print(f"  Mean signed diff: {diff_signed.mean().item():.4f}")
print(f"  Min signed diff:  {diff_signed.min().item():.4f}")
print(f"  Max signed diff:  {diff_signed.max().item():.4f}")
print(f"  % positive (eager > hip): {100 * (diff_signed > 0).float().mean().item():.1f}%")

# Check if it's a scale factor
ratio = out_eager.float() / (out_hip.float() + 1e-6)
print(f"  Mean ratio (eager/hip): {ratio.mean().item():.4f}")
print(f"  Median ratio: {ratio.median().item():.4f}")
