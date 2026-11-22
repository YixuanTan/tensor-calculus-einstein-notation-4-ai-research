# FlashAttention Backward Pass: Matrix Calculus vs. Einstein Notation Tensor Calculus

This repository provides an exact, step-by-step mathematical derivation of the **FlashAttention backward pass**, presented side by side in:
1. **Traditional Matrix Calculus** (whole-matrix transposes, traces, Hadamard products, trace cyclic permutations)
2. **Einstein Notation Tensor Calculus** (direct component contractions, Kronecker deltas, free index preservation)

---

## Interactive Comparison

An interactive, side-by-side comparison with expandable intermediate steps is available in [`matrix_vs_tensor_flash_attention.html`](matrix_vs_tensor_flash_attention.html) (or [`index.html`](index.html)).

### Key Features
- **Horizontally Aligned**: Each mathematical reasoning step (from output variation to softmax Jacobian, row-reduction identity $D$, and query/key gradients) is aligned side by side.
- **Collapsible Step Details**: Click any step to inspect every intermediate derivative without visual clutter.
- **Memory Map (HBM vs. SRAM)**: Explicit classification of formal inputs/outputs, auxiliary vectors ($\ell_i$, $D_i$), and on-chip transient tiles ($S, P, \bar{P}, \bar{S}$).

---

## Included Documents & Artifacts

- **`matrix_vs_tensor_flash_attention.html` / `index.html`**: Interactive web comparison.

---

## License

MIT License.
