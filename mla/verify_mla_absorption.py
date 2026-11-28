import torch


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding on the last dimension of x (..., L, ..., R)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def verify_mla_absorption_nope() -> None:
    """Verify exact equivalence of unfolded vs. absorbed MLA (nope branch)."""
    torch.manual_seed(0)
    B, L_q, L_kv = 2, 16, 32
    H, U, C, K, D = 4, 64, 24, 16, 16

    X_q = torch.randn(B, L_q, U, dtype=torch.float64)
    X_kv = torch.randn(B, L_kv, U, dtype=torch.float64)

    W_dkv = torch.randn(U, C, dtype=torch.float64) / (U ** 0.5)
    W_uk = torch.randn(C, H, K, dtype=torch.float64) / (C ** 0.5)
    W_uv = torch.randn(C, H, D, dtype=torch.float64) / (C ** 0.5)
    W_q = torch.randn(U, H, K, dtype=torch.float64) / (U ** 0.5)
    W_o = torch.randn(H, D, U, dtype=torch.float64) / ((H * D) ** 0.5)

    # 1. Unfolded (Training / Prefill)
    C_kv = torch.einsum("bju,uc->bjc", X_kv, W_dkv)
    K_nope = torch.einsum("bjc,chk->bjhk", C_kv, W_uk)
    V = torch.einsum("bjc,chd->bjhd", C_kv, W_uv)
    Q_nope = torch.einsum("biu,uhk->bihk", X_q, W_q)

    S_unfolded = torch.einsum("bihk,bjhk->bihj", Q_nope, K_nope) / (K ** 0.5)
    P_unfolded = torch.softmax(S_unfolded, dim=-1)
    O_unfolded = torch.einsum("bihj,bjhd,hdu->biu", P_unfolded, V, W_o)

    # 2. Absorbed (Decode): contract k and d offline into W_q_eff and W_o_eff
    W_q_eff = torch.einsum("uhk,chk->uhc", W_q, W_uk)
    W_o_eff = torch.einsum("chd,hdu->chu", W_uv, W_o)

    Q_tilde = torch.einsum("biu,uhc->bihc", X_q, W_q_eff)
    S_absorbed = torch.einsum("bihc,bjc->bihj", Q_tilde, C_kv) / (K ** 0.5)
    P_absorbed = torch.softmax(S_absorbed, dim=-1)

    O_latent = torch.einsum("bihj,bjc->bihc", P_absorbed, C_kv)
    O_absorbed = torch.einsum("bihc,chu->biu", O_latent, W_o_eff)

    assert torch.allclose(S_unfolded, S_absorbed, atol=1e-10)
    assert torch.allclose(O_unfolded, O_absorbed, atol=1e-10)
    print("✓ MLA nope absorption (W_uk -> W_q, W_uv -> W_o) verified")


def verify_mla_with_decoupled_rope() -> None:
    """Verify full MLA with Decoupled RoPE: nope absorbs, RoPE stays decoupled."""
    torch.manual_seed(42)
    B, L_q, L_kv = 2, 12, 20
    H, U, C, K, D, R = 4, 64, 24, 16, 16, 8

    X_q = torch.randn(B, L_q, U, dtype=torch.float64)
    X_kv = torch.randn(B, L_kv, U, dtype=torch.float64)

    W_dkv = torch.randn(U, C, dtype=torch.float64) / (U ** 0.5)
    W_uk = torch.randn(C, H, K, dtype=torch.float64) / (C ** 0.5)
    W_uv = torch.randn(C, H, D, dtype=torch.float64) / (C ** 0.5)
    W_q_nope = torch.randn(U, H, K, dtype=torch.float64) / (U ** 0.5)
    W_q_rope = torch.randn(U, H, R, dtype=torch.float64) / (U ** 0.5)
    W_k_rope = torch.randn(U, R, dtype=torch.float64) / (U ** 0.5)
    W_o = torch.randn(H, D, U, dtype=torch.float64) / ((H * D) ** 0.5)

    # Rotary frequencies
    angles_q = torch.randn(1, L_q, 1, R // 2, dtype=torch.float64)
    cos_q = torch.repeat_interleave(torch.cos(angles_q), 2, dim=-1)
    sin_q = torch.repeat_interleave(torch.sin(angles_q), 2, dim=-1)

    angles_kv = torch.randn(1, L_kv, R // 2, dtype=torch.float64)
    cos_kv = torch.repeat_interleave(torch.cos(angles_kv), 2, dim=-1)
    sin_kv = torch.repeat_interleave(torch.sin(angles_kv), 2, dim=-1)

    scale = (K + R) ** 0.5

    # 1. Unfolded (Training / Prefill)
    C_kv = torch.einsum("bju,uc->bjc", X_kv, W_dkv)
    K_nope = torch.einsum("bjc,chk->bjhk", C_kv, W_uk)
    V = torch.einsum("bjc,chd->bjhd", C_kv, W_uv)
    Q_nope = torch.einsum("biu,uhk->bihk", X_q, W_q_nope)

    Q_rope = apply_rope(torch.einsum("biu,uhr->bihr", X_q, W_q_rope), cos_q, sin_q)
    K_rope = apply_rope(torch.einsum("bju,ur->bjr", X_kv, W_k_rope), cos_kv, sin_kv)

    S_unfolded = (
        torch.einsum("bihk,bjhk->bihj", Q_nope, K_nope)
        + torch.einsum("bihr,bjr->bihj", Q_rope, K_rope)
    ) / scale
    P_unfolded = torch.softmax(S_unfolded, dim=-1)
    O_unfolded = torch.einsum("bihj,bjhd,hdu->biu", P_unfolded, V, W_o)

    # 2. Absorbed (Decode): only C_kv (bjc) and K_rope (bjr) are cached!
    W_q_eff = torch.einsum("uhk,chk->uhc", W_q_nope, W_uk)
    W_o_eff = torch.einsum("chd,hdu->chu", W_uv, W_o)

    Q_tilde = torch.einsum("biu,uhc->bihc", X_q, W_q_eff)
    S_absorbed = (
        torch.einsum("bihc,bjc->bihj", Q_tilde, C_kv)
        + torch.einsum("bihr,bjr->bihj", Q_rope, K_rope)
    ) / scale
    P_absorbed = torch.softmax(S_absorbed, dim=-1)
    O_absorbed = torch.einsum("bihj,bjc,chu->biu", P_absorbed, C_kv, W_o_eff)

    assert torch.allclose(S_unfolded, S_absorbed, atol=1e-10)
    assert torch.allclose(O_unfolded, O_absorbed, atol=1e-10)
    print("✓ Full MLA with Decoupled RoPE verified (KV cache = C_kv + K_rope only)")


if __name__ == "__main__":
    verify_mla_absorption_nope()
    verify_mla_with_decoupled_rope()

