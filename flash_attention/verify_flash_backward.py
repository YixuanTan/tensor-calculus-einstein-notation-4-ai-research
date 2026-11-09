"""
Verification script for FlashAttention Backward D_i identity:
D_matrix = sum(P * dP, dim=-1)   [Requires O(L^2) intermediate dP]
D_flash  = sum(O * dO, dim=-1)   [Contracted via associativity into O(L) online dot-product]
"""

import torch

def verify_flash_backward_identity():
    # Shapes: Batch=1, SeqLen=128, Dim=64
    B, L, D = 1, 128, 64
    torch.manual_seed(42)

    Q = torch.randn(B, L, D)
    K = torch.randn(B, L, D)
    V = torch.randn(B, L, D)

    # 1. Forward Pass
    # S_ij = (1/√D) Q_ik K_jk
    S = torch.einsum('bik,bjk->bij', Q, K) / (D ** 0.5)
    # P_ij = softmax_j(S_ij)
    P = torch.softmax(S, dim=-1)
    # O_id = P_ij V_jd
    O = torch.einsum('bij,bjd->bid', P, V)

    # 2. Simulated Upstream Gradient dO
    dO = torch.randn_like(O)

    # 3. Path A: Standard Matrix / Backprop (Instantiates intermediate dP)
    # dP_ij = dO_id V_jd
    dP = torch.einsum('bid,bjd->bij', dO, V)
    # D_i = sum_j P_ij * dP_ij  (Requires full L x L matrix in HBM)
    D_matrix = (P * dP).sum(dim=-1, keepdim=True)

    # 4. Path B: FlashAttention Tensor Contraction Identity
    # D_i = sum_d O_id * dO_id  (Only requires O and dO, zero intermediate P or dP needed!)
    D_flash = (O * dO).sum(dim=-1, keepdim=True)

    # 5. Assertion
    max_diff = (D_matrix - D_flash).abs().max().item()
    print(f"Max absolute discrepancy: {max_diff:.2e}")
    assert torch.allclose(D_matrix, D_flash, atol=1e-5), "Assertion failed: D_matrix != D_flash"
    print("✓ FlashAttention D_i = sum(O * dO) Identity Numerically Verified!")

if __name__ == "__main__":
    verify_flash_backward_identity()

