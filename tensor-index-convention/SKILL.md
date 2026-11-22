---
name: tensor-index-convention
description: >-
  Standardized Einstein summation and tensor index conventions for deep learning,
  attention mechanisms (MHA, GQA, MLA, FlashAttention), MoE/MLP blocks, adjoint
  backpropagation, and PyTorch einsum. Use when deriving tensor operations,
  writing attention or backpropagation math, avoiding transpose ambiguity in
  4D+ tensors, or translating derivations into torch.einsum.
---

# Tensor Index Convention

A clean, unambiguous Einstein summation convention for deep learning derivations. It eliminates 2D matrix transpose ambiguity ("transpose hell") and maps 1-to-1 to `torch.einsum`.

## 1. Axis & Index Assignments

Use single-character ASCII indices consistently to prevent name collisions across batch, head, sequence, feature, and routing dimensions:

| Dimension | Index | Range | Role & Convention |
| :--- | :---: | :--- | :--- |
| **Batch / Query Head** | $`b, h`$ | $`b \in [1, B],\, h \in [1, H]`$ | Free indices (suppressed in Stage A, restored in Stages B–C). |
| **KV Head Group** | $`g`$ | $`g \in [1, G]`$ | Key/Value head group index in GQA / MQA ($`G \le H`$). |
| **Query Sequence** | $`i`$ | $`i \in [1, L]`$ | Primary output sequence axis (free index). |
| **Key / Value Sequence** | $`j`$ | $`j \in [1, L]`$ | Contraction dummy index for attention weights. |
| **Normalization Sequence** | $`m, n`$ | $`m, n \in [1, L]`$ | Softmax / denominator dummy index. **Never reuse** $`j`$. |
| **Feature / Head Dim** | $`k`$ | $`k \in [1, D]`$ | Inner contraction channel (e.g., between $`Q`$ and $`K`$, or input features). |
| **Output / Value Dim** | $`d`$ | $`d \in [1, D_v]`$ | Value / output / residual-stream feature axis (free index). |
| **FFN Intermediate Dim** | $`f`$ | $`f \in [1, D_{\text{ff}}]`$ | Feed-forward (MLP / SwiGLU) hidden dimension. |
| **Latent / Expert / Rank** | $`c`$ | $`c \in [1, C]`$ | Compressed KV latent (MLA), MoE expert index, or LoRA rank. |

## 2. Notation & Summation Rules

**1. Component-as-Vector:** Do not add hats ($`\hat{Q}`$) or arrows ($`\vec{Q}`$) to denote vectors. Write $`Q_{ik}`$ (fixed $`i`$, free $`k`$) for the $`i`$-th token vector, or $`Q_i`$ in high-level schematics.

**2. Adjoint Sensitivity** ($`\bar{X}`$) **for Gradients:** Reserve the overbar exclusively for the adjoint sensitivity of a scalar loss $`\mathcal{L}`$ with respect to a tensor, $`\bar{X} \equiv \partial \mathcal{L} / \partial X`$. $`\bar{X}`$ always carries the **exact same index shape** as $`X`$ (e.g., $`\bar{O}_{id}`$, $`\bar{P}_{ij}`$, $`\bar{W}_{kd}`$).

**3. LHS Free Index Invariance (No Redundant Parentheses):** When an index appears as a free index on the left-hand side (LHS), it is automatically constrained as free on the right-hand side (RHS). **Do not** wrap it in parentheses or introduce Kronecker deltas on the RHS:

$$
P_{ij} = \frac{\exp(S_{ij})}{L_i}
$$

which maps directly to `torch.einsum('ij,i->ij', exp_S, 1 / L)`.

**4. Parenthesized Non-Summation Rule:** In **standalone expressions** without a constraining LHS, wrap any repeated index that must **not** be summed over in parentheses $`(\cdot)`$:

$$
D_{(i)}\, \bar{P}_{(i)j}, \qquad \frac{\exp(S_{(i)j})}{L_{(i)}}
$$

- With LHS: follow LHS free-index conservation (no parentheses).
- Without LHS: parenthesize non-contracted repeated indices.

**5. Explicit Broadcasting** with $`\mathbf{1}`$: Avoid implicit bias or scalar broadcasting. Multiply by an explicit all-ones vector $`\mathbf{1}`$ along the broadcast axis so backward reductions fall out mechanically:

$$
Y_{id} = X_{ik} W_{kd} + \mathbf{1}_i b_d \quad \Longrightarrow \quad \bar{b}_d = \mathbf{1}_i \bar{Y}_{id}
$$

## 3. Three-Stage Derivation Decomposition

Decompose multi-axis blocks into three stages for forward derivations, then reverse the stages when propagating adjoint sensitivities ($`\bar{X}`$):

- **Stage A — Local Pointwise / Single-Token Operations**: Suppress batch ($`b`$) and sequence ($`i`$) indices to derive token-wise blocks (projections, RMSNorm/LayerNorm, MLP/SwiGLU, RoPE, per-token MoE routing) cleanly.
- **Stage B — Cross-Position / Coupled Operations**: Reintroduce sequence ($`i, j, m, n`$) and head/group ($`h, g`$) indices for coupled layers (MHA, GQA/MQA, MLA, FlashAttention tiling, linear attention).
- **Stage C — Explicit Batch, Sequence, & Parameter Accumulation**: Restore all suppressed axes ($`b, h, i`$) to accumulate parameter gradients (e.g., $`\bar{W}_{kd} = X_{bik} \bar{Y}_{bid}`$) and cross-batch reductions.

## 4. Standard Matrix vs. Tensor Index Reference

| Operation | Standard Matrix / PyTorch | Einstein Tensor Index Notation |
| :--- | :--- | :--- |
| **Attention Forward** | `A = Q @ K.T / sqrt(d_k)`<br>`O = softmax(A) @ V` | $`S_{ij} = \frac{1}{\sqrt{d_k}} Q_{ik} K_{jk}`$<br>$`P_{ij} = \text{softmax}_j(S_{ij}),\quad O_{id} = P_{ij} V_{jd}`$ |
| **4D Production Tensors** `(B, H, L, D)` | `Q.transpose(1, 2) @ K.transpose(1, 2).transpose(-2, -1)` | $`S_{bhij} = \frac{1}{\sqrt{d_k}} Q_{bhik} K_{bhjk}`$<br>(Explicit axes; zero stride/perm bugs) |
| **GQA Attention** `(B, G, H/G, L, D)` | Reshape + repeat-interleave KV heads | $`S_{bghij} = \frac{1}{\sqrt{d_k}} Q_{bghik} K_{bgjk}`$ (with $`h \in [1, H/G]`$ per group $`g`$) |
| **Linear Layer + Bias** | `Y = X @ W + b` (implicit broadcast) | $`Y_{id} = X_{ik} W_{kd} + \mathbf{1}_i b_d,\quad \bar{b}_d = \mathbf{1}_i \bar{Y}_{id}`$ |
| **Linear Attention (Associativity)** | `(Q @ K.T) @ V` vs `Q @ (K.T @ V)` | $`O_{id} = Q_{ik} (K_{jk} V_{jd})`$ <br>(Contract $`j`$ first $`\to`$ state $`M_{kd}`$ in $`O(L D^2)`$) |
| **FlashAttention Backward Identity** | `D = rowsum(dP ⊙ P) = rowsum(dO ⊙ O)` | $`D_i = P_{im} \bar{P}_{im} = P_{im} (\bar{O}_{id} V_{md}) = (P_{im} V_{md}) \bar{O}_{id} = O_{id} \bar{O}_{id}`$ |

## 5. Verification with `torch.einsum`

When implementing or validating a derivation in code, translate each indexed equation directly into `torch.einsum` and verify numerically with `torch.allclose`.

