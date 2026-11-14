---
name: tensor-index-convention
description: >-
  Standardized Einstein summation and tensor index conventions for deep learning,
  attention mechanisms, and PyTorch einsum. Use when deriving tensor operations,
  writing attention or backpropagation math, avoiding transpose ambiguity in
  4D+ tensors, or translating derivations into torch.einsum.
---

# Tensor Index Convention

A clean, unambiguous Einstein summation convention for deep learning derivations. It eliminates 2D matrix transpose ambiguity ("transpose hell") and maps 1-to-1 to `torch.einsum`.

## 1. Axis & Index Assignments

Use consistent index letters to prevent name collisions across sequence, batch, and feature dimensions:

| Dimension | Index | Range | Role & Convention |
| :--- | :---: | :--- | :--- |
| **Batch / Head** | $b, h$ | $b \in [1, B],\, h \in [1, H]$ | Spectator free indices (or omit by setting $B=1, H=1$). |
| **Query Sequence** | $i$ | $i \in [1, L]$ | Primary output sequence axis (free index). |
| **Key / Value Sequence** | $j$ | $j \in [1, L]$ | Contraction dummy index for attention weights. |
| **Normalization Sequence** | $m, n$ | $m, n \in [1, L]$ | Softmax / denominator dummy index. **Never reuse $j$.** |
| **Feature / Head Dim** | $k$ | $k \in [1, D]$ | Inner contraction channel between $Q$ and $K$. |
| **Output Dim** | $d$ | $d \in [1, D_v]$ | Value / output feature axis (free index). |

## 2. Notation & Summation Rules

1. **Component-as-Vector**: Do not add hats ($\hat{Q}$) or bars ($\bar{Q}$) to denote vectors. Write $Q_{ik}$ (fixed $i$, free $k$) for the $i$-th token vector, or $Q_i$ in high-level schematics.
2. **LHS Free Index Invariance (No Redundant Parentheses)**:
   When an index appears as a free index on the left-hand side (LHS), it is automatically constrained as free on the right-hand side (RHS). **Do not** wrap it in parentheses on the RHS:
   $$P_{ij} = \frac{\exp(S_{ij})}{L_i} \quad \Longleftrightarrow \quad \text{\texttt{einsum('ij,i->ij', exp\_S, 1/L)}}$$
3. **Parenthesized Non-Summation Rule**:
   In **standalone expressions** without a constraining LHS, wrap any repeated index that must **not** be summed over in parentheses $(\cdot)$:
   $$D_{(i)}\, dP_{(i)j}, \qquad \frac{\exp(S_{(i)j})}{L_{(i)}}$$
   - With LHS: follow LHS free-index conservation (no parentheses).
   - Without LHS: parenthesize non-contracted repeated indices.

## 3. Standard Matrix vs. Tensor Index Reference

```text
Standard Matrix / PyTorch                Einstein Tensor Index Notation
───────────────────────────────────────  ───────────────────────────────────────
1. Attention Forward:
   A = Q @ K.T                           S_ij = (1/√d_k) Q_ik * K_jk
   O = A @ V                             P_ij = softmax_j(S_ij),  O_id = P_ij * V_jd

2. 4D Production Tensors (B, H, L, D):
   Q.transpose(1, 2) @                   Q_bhik * K_bhjk
   K.transpose(1, 2).transpose(-2, -1)   (Explicit axes; zero stride/perm bugs)

3. Linear Attention (Associative Law):
   O = (Q @ K.T) @ V  vs  Q @ (K.T @ V)  O_id = Q_ik * (K_jk * V_jd)
   (Error-prone transpose alignment)     Contract j first -> state S_kd in O(L D^2)

4. FlashAttention Backward Identity:
   D = rowsum(dP ⊙ P)                    D_i = P_im * dP_im
     = rowsum(dO ⊙ O)                        = P_im * (dO_id * V_md)
   (Requires matrix trace gymnastics)        = (P_im * V_md) * dO_id = O_id * dO_id
```

## 4. Verification with `torch.einsum`

Every symbolic derivation should translate directly into `torch.einsum` and be verified numerically with `torch.allclose`. See [verify_attention.py](./examples/verify_attention.py) for reference checks covering Linear Attention associativity and the FlashAttention $D_i = O_{id}\, dO_{id}$ identity.

