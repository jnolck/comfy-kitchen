import torch
from comfy_kitchen.backends.hip import quantize_int8_rowwise_convrot64
from comfy_kitchen.backends.hip import quantize_int8_rowwise
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation, _rotate_weight

torch.manual_seed(42)
M, K = 16, 256
device = "cuda"
dtype = torch.bfloat16
group_size = 256

x = torch.randn(M, K, dtype=dtype, device=device)

# HIP fused rotation + int8 quantize
qact_hip, scale_hip = quantize_int8_rowwise_convrot64(x, group_size)

# Eager: rotate then int8 quantize separately
h = _build_hadamard(group_size, device=device, dtype=dtype)
x_rot = _rotate_activation(x, h, group_size)
qact_eager, scale_eager = quantize_int8_rowwise(x_rot.contiguous())

print(f"qact match: {torch.equal(qact_hip, qact_eager)}")
print(f"qact max diff: {(qact_hip.int() - qact_eager.int()).abs().max().item()}")
print(f"scale match: {torch.allclose(scale_hip, scale_eager, atol=0.01)}")
print(f"scale max diff: {(scale_hip - scale_eager).abs().max().item():.6f}")
