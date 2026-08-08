import torch
import comfy_kitchen as ck

torch.manual_seed(42)
M, K, N = 32, 256, 128
device = "cuda"
dtype = torch.bfloat16

x = torch.randn(M, K, dtype=dtype, device=device)
weight_int4 = torch.randint(0, 256, (N, K // 2), dtype=torch.int32, device=device).to(torch.int8)
x_scales = torch.randn(M, dtype=torch.float32, device=device)
w_scales = torch.randn(N, dtype=torch.float32, device=device)

# Eager reference
with ck.use_backend("eager"):
    out_eager = ck.int8_linear(
        x, weight_int4, w_scales.unsqueeze(-1), out_dtype=dtype, convrot=False
    )

# HIP
with ck.use_backend("hip"):
    out_hip = ck.int8_linear(x, weight_int4, w_scales.unsqueeze(-1), out_dtype=dtype, convrot=False)

diff = (out_eager.float() - out_hip.float()).abs()
print(f"Max diff: {diff.max().item():.4f}")
print(f"Mean diff: {diff.mean().item():.4f}")
print(f"All close: {torch.allclose(out_eager.float(), out_hip.float(), atol=1.0)}")
print(f"\nFirst 8 eager: {out_eager[0, :8].float().tolist()}")
print(f"First 8 hip:   {out_hip[0, :8].float().tolist()}")
