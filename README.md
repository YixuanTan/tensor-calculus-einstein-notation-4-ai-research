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

### Topic 01: FlashAttention Backward Pass (`flash_attention/`)
Complete mathematical derivation of the FlashAttention backward kernel, comparing standard matrix calculus against Einstein notation side by side.

- **[Interactive Visualizer (`matrix_vs_tensor_flash_attention.html`)](flash_attention/matrix_vs_tensor_flash_attention.html)**: Clean, horizontally-aligned side-by-side comparison with expandable intermediate steps and HBM/SRAM memory maps.
- **[Complete Academic Derivation PDF](flash_attention/flash_attention_complete_derivation.pdf)**: 6-page comprehensive document detailing variations, the softmax Jacobian, the row-reduction invariant $D_i = O_{id}\bar{O}_{id}$, and hardware tiling.

---

## 🗺️ Roadmap / Upcoming Topics

- [x] **01. FlashAttention**: Exact backward pass, row reduction $D_i$, and on-chip SRAM lifecycle.
- [ ] **02. Rotary Position Embeddings (RoPE)**: Tensor contractions, complex vector space isomorphism, and head-wise rotations.
- [ ] **03. Normalization Layers**: LayerNorm and RMSNorm adjoint derivations in index notation.
- [ ] **04. GQA & MQA**: Grouped-query and multi-query attention tensor projections and memory reuse.
- [ ] **05. SwiGLU & Feedforward Blocks**: Fused elementwise gate gradients in Einstein notation.

---

## 📄 License

MIT License.
