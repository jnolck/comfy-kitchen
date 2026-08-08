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


def decode_nvfp4_e2m1(nibble: int) -> float:
    """NVFP4 E2M1: 1 sign, 2 exp, 1 mant, bias=1"""
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1

    if exp == 0:
        val = mant * 0.5
    elif exp == 1:
        val = 1.0 + mant * 0.5
    elif exp == 2:
        val = 2.0 + mant * 1.0
    else:
        val = 4.0 + mant * 2.0

    return -val if sign else val


# Test single known byte with single block
# NVFP4 block: 16 values per block, 2 bytes per 4 values = 8 bytes per block
# Let's encode all 16 nibble values in order: 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF
# HiFirst: high nibble = even index, low nibble = odd index
# So: val0=0, val1=1, val2=2, val3=3, val4=4, val5=5, val6=6, val7=7,
#     val8=8, val9=9, val10=A, val11=B, val12=C, val13=D, val14=E, val15=F
qx = torch.zeros(16, 16, device=device, dtype=torch.uint8)  # 16 rows, 8 bytes = 16 values
qx[0, 0] = 0x10
qx[0, 1] = 0x32
qx[0, 2] = 0x54
qx[0, 3] = 0x76
qx[0, 4] = 0x98
qx[0, 5] = 0xBA
qx[0, 6] = 0xDC
qx[0, 7] = 0xFE

# Check what the kernel sees as the first decoded value
print(f"Byte 0: 0x{qx[0, 0].item():02x}")
print(f"  High nibble: {qx[0, 0].item() >> 4} (= {decode_nvfp4_e2m1(qx[0, 0].item() >> 4)})")
print(f"  Low nibble: {qx[0, 0].item() & 0x0F} (= {decode_nvfp4_e2m1(qx[0, 0].item() & 0x0F)})")
print(
    f"  HiFirst means val0=high={decode_nvfp4_e2m1(qx[0, 0].item() >> 4)}, val1=low={decode_nvfp4_e2m1(qx[0, 0].item() & 0x0F)}"
)

per_tensor_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
block_scales = torch.ones(16, 1, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

# Print the raw bytes of the block scale tensor
raw_scale = block_scales.view(torch.uint8)
print(f"Block scale raw byte: 0x{raw_scale[0, 0].item():02x}")

from comfy_kitchen.backends.hip import dequantize_nvfp4

result = dequantize_nvfp4(qx, per_tensor_scale, block_scales, torch.float32)

print("All 16 NVFP4 nibbles decoded with scale=1.0:")
for i in range(16):
    expected = decode_nvfp4_e2m1(i)
    got = result[0, i].item()
    match = "✓" if abs(got - expected) < 0.01 else "✗"
    print(f"  nibble 0x{i:01X}: expected={expected:6.2f}, got={got:6.2f} {match}")
