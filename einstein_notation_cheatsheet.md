# Einstein Notation for AI & LLMs — Beginner's Cheatsheet

> **Core Mental Model:** Einstein notation is a **mathematical writing & derivation system** for tensors of any rank. By writing each tensor as a single indexed scalar, ordinary scalar algebra replaces transposes, $`\mathrm{diag}`$, $`\odot`$, and $`\mathrm{Tr}`$ — and scales effortlessly from 2D matrices to 4D+ attention tensors.

---

## 1. The 3 Core Rules

| Rule | Meaning | Example |
| :--- | :--- | :--- |
| **1. Scalar View** | Every tensor is written as a single real number at specific coordinates. | $`Q_{bhik} \in \mathbb{R}`$ |
| **2. Free vs. Dummy** | **Free index** = appears on the left-hand side (LHS) $`\implies`$ kept in output, **never summed**.<br>**Dummy index** = repeated on the RHS *only* $`\implies`$ **automatically summed** (contracted). | $`C_{ij} = A_{ik} B_{kj} \equiv \sum_k A_{ik} B_{kj}`$ |
| **3. Commutativity** | Because components are plain scalars, factors commute freely — **transposes vanish**. | $`Q_{ik} K_{jk} = K_{jk} Q_{ik}`$ (no $`K^\top`$!) |

---

## 2. Standard AI & LLM Axis Dictionary

Use consistent letters so indices never collide across equations:

| Letter | Meaning | Typical Role |
| :---: | :--- | :--- |
| $`b, h`$ | Batch, Attention Head | **Free** (carried unchanged across batch & head) |
| $`i`$ | Query sequence position | **Free** (output row / token) |
| $`j`$ | Key / Value sequence position | **Free** (in $`S_{ij}`$) or **Dummy** (in $`O_{id} = P_{ij} V_{jd}`$) |
| $`k`$ | Query–Key head dimension ($`D_k`$) | **Dummy** (contracted between $`Q`$ and $`K`$) |
| $`d`$ | Value / Output feature dimension ($`D_v`$) | **Free** (output feature axis) |
| $`m, n`$ | Inner sums (e.g., Softmax denominator) | **Dummy** — *never reuse* $`j`$ |

---

## 3. Rosetta Stone: Matrix Calculus ↔ Einstein Notation

| Operation | Matrix Calculus | Einstein Notation | Explicit Meaning |
| :--- | :--- | :--- | :--- |
| **Dot product** | $`s = u^\top v`$ | $`s = u_k v_k`$ | $`\sum_k u_k v_k`$ |
| **Outer product** | $`M = u v^\top`$ | $`M_{ij} = u_i v_j`$ | $`(u_i)(v_j)`$ (no sum) |
| **Matrix–vector** | $`y = A x`$ | $`y_i = A_{ij} x_j`$ | $`\sum_j A_{ij} x_j`$ |
| **Matrix multiply** | $`C = A B`$ | $`C_{ij} = A_{ik} B_{kj}`$ | $`\sum_k A_{ik} B_{kj}`$ |
| **Transpose** | $`B = A^\top`$ | $`B_{ji} = A_{ij}`$ | swap index order |
| **Attention scores (2D)** | $`S = Q K^\top`$ | $`S_{ij} = Q_{ik} K_{jk}`$ | $`\sum_k Q_{ik} K_{jk}`$ (no transpose!) |
| **4D Multi-Head scores** | 2D notation breaks | $`S_{bhij} = Q_{bhik} K_{bhjk}`$ | $`\sum_k Q_{bhik} K_{bhjk}`$ |
| **4D Attention output** | 2D notation breaks | $`O_{bhid} = P_{bhij} V_{bhjd}`$ | $`\sum_j P_{bhij} V_{bhjd}`$ |
| **Row sum** | $`r = A \mathbf{1}`$ | $`r_i = \sum_j A_{ij}`$ | sum over $`j`$ |
| **Row scaling** | $`Y = \mathrm{diag}(a) X`$ | $`Y_{ij} = a_i X_{ij}`$ | $`i`$ on LHS $`\implies`$ not summed |
| **Elementwise (Hadamard)** | $`C = A \odot B`$ | $`C_{ij} = A_{ij} B_{ij}`$ | $`i, j`$ on LHS $`\implies`$ not summed |
| **Extract diagonal** | $`v = \mathrm{diag}(A)`$ | $`v_i = A_{ii}`$ | $`i`$ on LHS $`\implies`$ not summed |
| **Trace** | $`t = \mathrm{Tr}(A)`$ | $`t = A_{ii}`$ | $`\sum_i A_{ii}`$ |
| **Frobenius inner product** | $`s = \mathrm{Tr}(A^\top B)`$ | $`s = A_{ij} B_{ij}`$ | $`\sum_{i,j} A_{ij} B_{ij}`$ |

---

## 4. The 3 Common Beginner Pitfalls

### Pitfall 1: “A letter repeats on the RHS — do I sum it?”
**Always check the left-hand side (LHS) first!**

**Present on LHS** $`\implies`$ **NOT summed (free index):**

$$
P_{ij} = \frac{\exp(S_{ij})}{\ell_i} \qquad (i \text{ is on the LHS, so do NOT sum } i)
$$

$$
D_i = P_{im} \bar{P}_{im} \qquad (i \text{ is free; } m \text{ is summed})
$$

**No LHS (standalone expression)** $`\implies`$ **wrap non-summed repeated indices in parentheses:**

$$
D_{(i)} \bar{P}_{(i)j}
$$

### Pitfall 2: Reusing a free index for an inner sum
Never use the same letter for both a free output index and a summation dummy index:

- **Bad:** $`\ell_i = \sum_j \exp(S_{ij})`$ inside $`P_{ij} = \exp(S_{ij}) / \sum_j \exp(S_{ij})`$
- **Good:** Use a fresh dummy letter $`m`$ or $`n`$:

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_m \exp(S_{im})}
$$

### Pitfall 3: Treating it as mere shorthand instead of a derivation system
The real payoff of Einstein notation is **algebraic manipulation on paper** — because every term is a plain scalar, you can freely regroup and commute factors:

**Linear Attention** ($`O(L^2 D) \to O(L D^2)`$) — just scalar associativity:

$$
O_{id} = \underbrace{(Q_{ik} K_{jk})}_{O(L^2 D)} V_{jd} = Q_{ik} \underbrace{(K_{jk} V_{jd})}_{O(L D^2)}
$$

**FlashAttention Backward Identity** — 1 line instead of matrix trace tricks:

$$
D_i = P_{im} \bar{P}_{im} = P_{im} (\bar{O}_{id} V_{md}) = (P_{im} V_{md}) \bar{O}_{id} = O_{id} \bar{O}_{id}
$$

---

## 5. Advanced Starter: Kronecker Delta & Backprop

**Definition:** $`\delta_{ij} = 1`$ if $`i = j`$, else $`0`$ (the identity matrix in index form).

**Index renamer:** Multiplying by $`\delta_{ij}`$ replaces the summed index:

$$
\delta_{ij} v_j = v_i
$$

**Where** $`\delta`$ **comes from in partial derivatives:** Because $`S_{i1}, \dots, S_{iL}`$ are independent coordinates in row $`i`$, there are two cases:
- **If** $`j = m`$: $`S_{im}`$ and $`S_{ij}`$ are the same variable, so $`\dfrac{\partial S_{im}}{\partial S_{ij}} = 1`$.
- **If** $`j \neq m`$: $`S_{im}`$ does not depend on $`S_{ij}`$, so $`\dfrac{\partial S_{im}}{\partial S_{ij}} = 0`$.

Combining both cases via the Kronecker delta ($`\delta_{jm} = 1`$ if $`j = m`$, else $`0`$):

$$
\frac{\partial S_{im}}{\partial S_{ij}} = \delta_{jm} \qquad \Big(\text{and in general: } \frac{\partial x_i}{\partial x_j} = \delta_{ij}, \quad \frac{\partial A_{pq}}{\partial A_{ij}} = \delta_{pi}\delta_{qj}\Big)
$$

**1-line matrix differentiation** (differentiate $`\to \delta \to`$ rename index):

$$
\frac{\partial (W_{ik} x_k)}{\partial x_j} = W_{ik} \frac{\partial x_k}{\partial x_j} = W_{ik} \delta_{kj} = W_{ij}
$$

**Softmax Jacobian** (no matrix tricks):

$$
\frac{\partial P_{ij}}{\partial S_{im}} = P_{ij}(\delta_{jm} - P_{im})
$$

---

## 6. 3-Step Workflow for Any Paper or Algorithm

1. **Represent:** Write every tensor with explicit axes ($`b, h, i, j, k, d`$).
2. **Derive:** Regroup and commute sums like ordinary scalar algebra.
3. **Differentiate:** Apply the component-wise chain rule with $`\delta_{ij}`$ — no matrix Jacobians required.
