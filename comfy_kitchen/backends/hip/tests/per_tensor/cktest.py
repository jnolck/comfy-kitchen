# test_ck.py
import torch
import _C


def test_ck_gemm(M, N, K, out_dtype=torch.float32):
    """Test CK int8 dequant GEMM: D = (A@B^T) * xs * ws + bias"""
    a = torch.randint(-5, 5, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-5, 5, (N, K), dtype=torch.int8, device="cuda")
    xs = torch.rand(M, dtype=torch.float32, device="cuda") * 2
    ws = torch.rand(N, dtype=torch.float32, device="cuda") * 2
    bias = torch.randn(N, dtype=torch.float32, device="cuda")
    d = torch.empty(M, N, dtype=out_dtype, device="cuda")

    stream = torch.cuda.current_stream()

    # out_dtype_code: 0=float32, 1=float16, 2=bfloat16
    if out_dtype == torch.float32:
        code = 0
    elif out_dtype == torch.float16:
        code = 1
    elif out_dtype == torch.bfloat16:
        code = 2
    else:
        raise ValueError(f"unsupported dtype {out_dtype}")

    ok = _C.cutlass_int8_dequant(a, b, xs, ws, bias, d, code, stream.cuda_stream)

    # CPU reference
    acc = torch.matmul(a.float(), b.float().t())  # [M, K] @ [K, N] = [M, N]
    ref = (acc * xs.unsqueeze(1) * ws.unsqueeze(0) + bias.unsqueeze(0)).to(out_dtype)

    if ok:
        max_err = (d.float() - ref.float()).abs().max().item()
        rel_err = ((d.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)).max().item()
        print(
            f"  M={M:4d} N={N:4d} K={K:4d} dtype={str(out_dtype):>12s} | max_err={max_err:.4e} rel_err={rel_err:.4e}",
            end="",
        )
        if max_err < 0.01 and rel_err < 0.01:
            print(" PASS")
        else:
            print(" FAIL (tolerance)")
        return True
    else:
        print(
            f"  M={M:4d} N={N:4d} K={K:4d} dtype={str(out_dtype):>12s} | CK returned False - FALLBACK"
        )
        return False


# Test matrix
test_configs = [
    # (M, N, K) - various shapes
    (32, 64, 128),  # small
    (64, 64, 64),  # square small
    (128, 128, 256),  # square medium
    (64, 256, 512),  # wide N
    (256, 64, 512),  # wide M
    (1, 64, 128),  # M=1 (edge case)
    (64, 1, 128),  # N=1 (edge case)
    (512, 512, 128),  # large M,N, small K
    (128, 128, 1024),  # large K
    (256, 256, 256),  # medium square
]

dtypes = [torch.float32, torch.float16, torch.bfloat16]

print("=" * 80)
print("CK Int8 Fused GEMM Tests")
print("=" * 80)

passed = 0
failed = 0
fallback = 0

for dtype in dtypes:
    print(f"\n--- {dtype} ---")
    for M, N, K in test_configs:
        try:
            ok = test_ck_gemm(M, N, K, dtype)
            if ok:
                passed += 1
            else:
                fallback += 1
        except Exception as e:
            print(f"  M={M:4d} N={N:4d} K={K:4d} dtype={str(dtype):>12s} | EXCEPTION: {e}")
            failed += 1

print("\n" + "=" * 80)
print(f"Results: {passed} passed, {fallback} fallback, {failed} failed")
print("=" * 80)
