# Tensor Calculus and Einstein Notation for AI Research

A systematic collection of mathematical derivations and visual comparisons applying **Einstein notation tensor calculus** to modern deep learning architectures and hardware-aware operators.

---

## 🎯 Motivation

Traditional machine learning literature relies heavily on **matrix calculus** (whole-matrix transposes, traces, Hadamard products, and trace cyclic permutations). While suitable for shallow 2D operations, matrix calculus quickly becomes cumbersome, opaque, and error-prone when dealing with:
- Multi-dimensional attention tensors (batch, head, sequence, hidden dimension)
- Hardware memory hierarchy mappings (SRAM vs. HBM tiling)
- Backpropagation through nonlinear elementwise reductions and softmax

By contrast, **Einstein Notation Tensor Calculus** (used extensively in continuum mechanics and theoretical physics):
1. **Preserves Index Semantics**: Every index ($i, j, k, d$) has an invariant geometric/hardware meaning.
2. **Eliminates Ad-Hoc Rules**: Chain rules and product rules apply directly at the scalar component level; contractions occur naturally via repeated indices.
3. **Maps Directly to Kernels**: The index expression translates 1-to-1 into GPU thread indexing and loop scheduling (e.g. Triton, CUDA, CUTLASS).

---

## 📚 Topics & Repository Structure

### Topic 00: Foundations — Einstein Notation for AI & LLMs (`einstein_basics/`)
A 10-minute beginner's guide to Einstein notation for deep learning (3 minimal rules, Kronecker delta backpropagation, Linear Attention, and an open-source Agent Skill).

- **[English Blog Post (Web)](https://yixuantan.github.io/tensor-calculus-einstein-notation-4-ai-research/einstein_basics/blog/einstein_notation_for_ai.html)** ([Markdown](einstein_basics/blog/einstein_notation_for_ai.md)): Complete walkthrough from 1D dot products to 4D attention and $\delta_{ij}$ differentiation.
- **[Interactive Slides & Playground (`einstein_notation_intro.html`)](https://yixuantan.github.io/tensor-calculus-einstein-notation-4-ai-research/einstein_basics/einstein_notation_intro.html)**: Live interactive index highlighter, Kronecker delta visualizer, and Linear Attention complexity slider.
- **[1-Page Beginner's Cheatsheet (`einstein_notation_cheatsheet.md`)](einstein_basics/einstein_notation_cheatsheet.md)**: Rosetta Stone, pitfalls, and Kronecker delta backprop quick reference.
- **[Agent Skill (`tensor-index-convention/SKILL.md`)](tensor-index-convention/SKILL.md)**: Drop-in skill for Claude Code / Cursor / Antigravity to make AI derive clean index notation & `torch.einsum`.

### Topic 01: FlashAttention Backward Pass (`flash_attention/`)
Complete mathematical derivation of the FlashAttention backward kernel, comparing standard matrix calculus against Einstein notation side by side.

- **[Interactive Visualizer (`matrix_vs_tensor_flash_attention.html`)](flash_attention/matrix_vs_tensor_flash_attention.html)**: Clean, horizontally-aligned side-by-side comparison with expandable intermediate steps and HBM/SRAM memory maps.

### Topic 02: Sparse Attention Backward Pass (`sparse_attention/`)
Interactive and mathematical comparison showing why $\bar{Q}$ sweeps rows (CSR), $\bar{K}/\bar{V}$ sweep columns (CSC), and how index notation reveals `atomicAdd` write conflicts hidden by matrix transposes.

- **[Interactive Row-vs-Column Visualizer (`sparse_attention_rows_vs_columns.html`)](sparse_attention/sparse_attention_rows_vs_columns.html)**: Interactive $6 \times 6$ sparse mask grid demonstrating row sweeps (CSR), atomic write contention (`atomicAdd`), and lock-free column sweeps (CSC).

---

## 🗺️ Roadmap / Upcoming Topics

- [x] **01. FlashAttention**: Exact backward pass, row reduction $D_i$, and on-chip SRAM lifecycle.
- [x] **02. Sparse Attention**: Masking $M_{ij}$, CSR vs. CSC schedules, and hardware write-conflict analysis.
- [ ] **03. Rotary Position Embeddings (RoPE)**: Tensor contractions, complex vector space isomorphism, and head-wise rotations.
- [ ] **04. Normalization Layers**: LayerNorm and RMSNorm adjoint derivations in index notation.
- [ ] **05. GQA & MQA**: Grouped-query and multi-query attention tensor projections and memory reuse.
- [ ] **06. SwiGLU & Feedforward Blocks**: Fused elementwise gate gradients in Einstein notation.

---

## 📄 License

MIT License.
