import torch


def verify_linear_attention_associative_law() -> None:
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 256, 64
    Q = torch.randn(B, H, L, D, dtype=torch.float64)
    K = torch.randn(B, H, L, D, dtype=torch.float64)
    V = torch.randn(B, H, L, D, dtype=torch.float64)

    phi_Q = torch.nn.functional.elu(Q) + 1.0
    phi_K = torch.nn.functional.elu(K) + 1.0

    # Left-associative: O(L^2 D)
    A = torch.einsum("bhik,bhjk->bhij", phi_Q, phi_K)
    O_quad = torch.einsum("bhij,bhjd->bhid", A, V)

    # Right-associative (contract j first): O(L D^2)
    S = torch.einsum("bhjk,bhjd->bhkd", phi_K, V)
    O_linear = torch.einsum("bhik,bhkd->bhid", phi_Q, S)

    assert torch.allclose(O_quad, O_linear, atol=1e-8)
    print("✓ Linear Attention associative law verified")


def verify_flash_backward_identity() -> None:
    B, L, D = 1, 64, 32
    Q = torch.randn(B, L, D)
    K = torch.randn(B, L, D)
    V = torch.randn(B, L, D)

    S = torch.einsum("bik,bjk->bij", Q, K) / (D ** 0.5)
    P = torch.softmax(S, dim=-1)
    O = torch.einsum("bij,bjd->bid", P, V)
    dO = torch.randn_like(O)

    # D_i = P_im * dP_im == O_id * dO_id
    dP = torch.einsum("bid,bjd->bij", dO, V)
    D_standard = torch.einsum("bim,bim->bi", P, dP)
    D_flash = torch.einsum("bid,bid->bi", O, dO)

    assert torch.allclose(D_standard, D_flash, atol=1e-5)
    print("✓ FlashAttention D_i = O_id * dO_id identity verified")


if __name__ == "__main__":
    verify_linear_attention_associative_law()
    verify_flash_backward_identity()
