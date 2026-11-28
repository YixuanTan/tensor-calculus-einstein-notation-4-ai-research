# Multi-Head Latent Attention (MLA): Matrix Calculus vs. Einstein Notation

> Follows the [Tensor Index Convention](../tensor-index-convention/SKILL.md):
> - $b$: batch (optional spectator index, dropped below for clarity)
> - $h \in [1, H]$: attention head ($H = 128$ in DeepSeek-V2)
> - $i \in [1, L_q]$: query sequence position ($L_q = 1$ at decode)
> - $j \in [1, L_{kv}]$: key/value history position ($L_{kv} = L$, e.g., $100\text{k}$)
> - $u \in [1, D_{\text{model}}]$: model hidden dimension
> - $c \in [1, D_c]$: compressed KV latent dimension ($D_c = 512 \ll H D_k$)
> - $k, k' \in [1, D_k^{\text{nope}}],\; d \in [1, D_v]$: per-head nope key / value feature dimensions ($128$)
> - $r, r' \in [1, D_k^{\text{rope}}],\; m$: pre-RoPE and post-RoPE feature dimensions ($D_k^{\text{rope}} = 64$)

---

## Showdown 1: Multi-Head Latent Compression (2D Matrix Collapse vs. Native Tensor Axes)

| Traditional Matrix Calculus | Einstein Notation Tensor Calculus |
| :--- | :--- |
| A 2D matrix only has **2 slots** (rows & cols), so $(j, h, c, k)$ cannot fit at once. Papers either flatten heads into $L \times H D_k$ and slice, or drop sequence $L$ entirely and write single-token column vectors ($t$): | Every axis gets its own permanent slot—all $L$ tokens and all $H$ heads coexist in one flat formula with **zero reshapes or slices**: |
| $$C = X W^{\text{DKV}} \in \mathbb{R}^{L \times D_c}$$ | $$C_{jc} = X_{ju}\, W^{\text{DKV}}_{uc} \quad (\text{no } h!)$$ |
| $$K_{\text{cat}} = C W^{\text{UK}} \in \mathbb{R}^{L \times H D_k},\quad K^{(h)} = \text{slice}_h(K_{\text{cat}})$$ | $$K^{\text{nope}}_{jhk} = C_{jc}\, W^{\text{UK}}_{chk}$$ |
| $$V_{\text{cat}} = C W^{\text{UV}} \in \mathbb{R}^{L \times H D_v},\quad V^{(h)} = \text{slice}_h(V_{\text{cat}})$$ | $$V_{jhd} = C_{jc}\, W^{\text{UV}}_{chd}$$ |
| $$Q_{\text{cat}} = X W^{\text{Q}} \in \mathbb{R}^{L \times H D_k},\quad Q^{(h)} = \text{slice}_h(Q_{\text{cat}})$$ | $$Q^{\text{nope}}_{ihk} = X_{iu}\, W^{\text{Q,nope}}_{uhk}$$ |

---

## Showdown 2: Key & Value Absorption (Where 2D Matrix Associativity Actually Gives the Wrong Answer!)

### 2A. Key Absorption ($W^{\text{UK}} \to W^{\text{Q}}$)

| Traditional Matrix Calculus | Einstein Notation Tensor Calculus |
| :--- | :--- |
| **Trap!** Naive 2D matrix associativity on full concatenated weights $W^{\text{Q}}, W^{\text{UK}} \in \mathbb{R}^{(\cdot) \times H D_k}$ is **WRONG**: | Associativity on dummy index $k$—**LHS free-index rule makes the bug impossible**: |
| $$(X W^{\text{Q}})(C W^{\text{UK}})^\top = X\big(W^{\text{Q}}(W^{\text{UK}})^\top\big)C^\top \;\; \color{#ff4d4f}{\text{✗ sums across all } h!}$$ | $$S^{\text{nope}}_{i\color{#1677ff}{h}j} = Q^{\text{nope}}_{i\color{#1677ff}{h}k}\, K^{\text{nope}}_{j\color{#1677ff}{h}k} = \big(X_{iu}\, W^{\text{Q,nope}}_{u\color{#1677ff}{h}k}\big)\big(C_{jc}\, W^{\text{UK}}_{c\color{#1677ff}{h}k}\big)$$ |
| Forced to manually slice per head, transpose each slice, and track $H$ separate matrix products: | $$= X_{iu}\, \underbrace{\big(W^{\text{Q,nope}}_{u\color{#1677ff}{h}k}\, W^{\text{UK}}_{c\color{#1677ff}{h}k}\big)}_{W^{\text{Q,eff}}_{u\color{#1677ff}{h}c}}\, C_{jc} = \tilde{Q}_{i\color{#1677ff}{h}c}\, C_{jc}$$ |
| $$S^{(h)} = X \Big( W^{\text{Q}}_{(h)} \big(W^{\text{UK}}_{(h)}\big)^\top \Big) C^\top$$ | **$h$ is on the LHS** $\implies$ never summed! Only $k$ contracts offline; **$C_{jc}$ carries no $h$**. |

### 2B. Value Absorption ($W^{\text{UV}} \to W^{\text{O}}$)

| Traditional Matrix Calculus | Einstein Notation Tensor Calculus |
| :--- | :--- |
| Now $h$ **must** be summed—forced to write an explicit $\sum_{h=1}^H$ loop over head slices: | Same single line; **$h$ is absent from LHS $O_{iu}$**, so $h$ contracts automatically at the exact right step: |
| $$O = \sum_{h=1}^{H} \big(P^{(h)} C W^{\text{UV}}_{(h)}\big) W^{\text{O}}_{(h)} = \sum_{h=1}^{H} \big(P^{(h)} C\big)\big(W^{\text{UV}}_{(h)} W^{\text{O}}_{(h)}\big)$$ | $$O_{iu} = P_{i\color{#fa8c16}{h}j}\big(C_{jc}\, W^{\text{UV}}_{c\color{#fa8c16}{h}d}\big) W^{\text{O}}_{\color{#fa8c16}{h}du} = \underbrace{\big(P_{i\color{#fa8c16}{h}j}\, C_{jc}\big)}_{\tilde{O}_{i\color{#fa8c16}{h}c}}\, \underbrace{\big(W^{\text{UV}}_{c\color{#fa8c16}{h}d}\, W^{\text{O}}_{\color{#fa8c16}{h}du}\big)}_{W^{\text{O,eff}}_{c\color{#fa8c16}{h}u}}$$ |
| Easy to wonder whether $P^{(h)} C$ can be summed over $h$ early. | **Auto bug-guard**: both $\tilde{O}_{ihc}$ and $W^{\text{O,eff}}_{chu}$ carry $h$, so $h$ contracts pairwise; $V_{jhd}$ is never materialized. |

> **Production Note (Offline Merging vs. $L_q=1$ Online Two-Step `bmm`):**
> Materializing $W^{\text{O,eff}}_{chu}$ offline ($512 \times 128 \times 5120 \approx 335\text{M}$ params/layer) adds **~40 GB of static weight VRAM** across 60 layers in BF16 (while Prefill still requires unmerged weights). Production engines (vLLM / newer SGLang) therefore often keep $W^{\text{UV}}, W^{\text{O}}$ unmerged and execute the same associative order **online** for $L_q = 1$: $\big((P_{ihj} C_{jc}) W^{\text{UV}}_{chd}\big) W^{\text{O}}_{hdu}$ (and $\big((X_{iu} W^{\text{Q,nope}}_{uhk}) W^{\text{UK}}_{chk}\big) C_{jc}$)—still never materializing $K_{jhk}$ or $V_{jhd}$, with zero extra static weight memory.

---

## Showdown 3: Why RoPE Breaks Absorption & Why Decoupled RoPE Is Inevitable

| Traditional Matrix Calculus | Einstein Notation Tensor Calculus |
| :--- | :--- |
| Cannot express per-row position rotation on a whole matrix $Q \in \mathbb{R}^{L \times D}$ as a 2D matrix multiply—forced to hide it inside $\mathrm{RoPE}(\cdot)$ or drop back to single-token pairs: | RoPE is a first-class 3D tensor $R_{pmk}$ (position $p$, rotated dim $m$, input dim $k$), on the exact same footing as $X, W, C$: |
| $$S^{(h)}_{ij} = x_i^\top W^{\text{Q}}_{(h)} \mathbf{R}_i^\top \mathbf{R}_j \big(W^{\text{UK}}_{(h)}\big)^\top c_j$$ | $$Q^{\text{rot}}_{ihm} = R_{imk}\, X_{iu}\, W^{\text{Q}}_{uhk}, \qquad K^{\text{rot}}_{jhm} = R_{jmk'}\, C_{jc}\, W^{\text{UK}}_{chk'}$$ |
| $\mathbf{R}_i^\top$ and $\mathbf{R}_j$ look symmetric, hiding both the $L_q = 1 \text{ vs. } L_{kv} = 100\text{k}$ asymmetry and the $h$-vs.-$j$ clash. | $$S^{\text{coupled}}_{ihj} = Q^{\text{rot}}_{ihm}\, K^{\text{rot}}_{jhm} = X_{iu}\, W^{\text{Q}}_{uhk}\, \big(R_{imk}\, R_{jmk'}\big)\, W^{\text{UK}}_{chk'}\, C_{jc}$$ |

> **Step-by-step derivation of the middle bracket $R_{imk}\, R_{jmk'} = R_{(j-i),\, kk'}$ (zero skipped steps):**
> 1. **Avoid dummy-index collision**: Pre-RoPE $Q$ contracts over $k$, while pre-RoPE $K$ uses a separate dummy index $k'$ so multiplying $Q^{\text{rot}}_{ihm} K^{\text{rot}}_{jhm}$ doesn't repeat $k$ four times.
> 2. **Scalar reordering**: Since every indexed component is a plain real scalar, pull $R_{imk}$ and $R_{jmk'}$ together; only they carry $m$ (which is absent from LHS $S_{ihj}$), so $m$ sums out inside the bracket.
> 3. **Why $\mathbf{R}_i^\top$ appears**: Fixing $i, j$, $R_{imk} = (\mathbf{R}_i)_{mk}$. Summing over the **first** slot $m$ of $(\mathbf{R}_i)_{mk}$ and $(\mathbf{R}_j)_{mk'}$ matches standard matrix multiplication once we transpose the first factor: $\sum_m (\mathbf{R}_i^\top)_{km} (\mathbf{R}_j)_{mk'} = (\mathbf{R}_i^\top \mathbf{R}_j)_{kk'}$.
> 4. **Where $(j-i)$ comes from**: Each $2\times 2$ block of $\mathbf{R}_p$ rotates by angle $p\theta$, $\begin{pmatrix}\cos(p\theta) & -\sin(p\theta) \\ \sin(p\theta) & \cos(p\theta)\end{pmatrix}$. Transposing flips the $\sin$ signs ($\mathbf{R}_i^\top = \mathbf{R}_{-i}$, rotating back by $-i\theta$), and composing rotations adds angles: $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{-i}\mathbf{R}_j = \mathbf{R}_{j-i}$. Thus $\big(\mathbf{R}_i^\top \mathbf{R}_j\big)_{kk'} = \big(\mathbf{R}_{j-i}\big)_{kk'} \equiv R_{(j-i),\, kk'}$ (the same rotation tensor $R_{p,\,kk'}$ evaluated at relative position $p = j - i$).

### The $h$-vs.-$j$ Trap (Why $R_j$—Not $R_i$—Kills Absorption at Decode)

At Decode ($L_q = 1$ new token $i$ vs. $L_{kv} = 100\text{k}$ cached tokens $j$), rotating the single current query $Q^{\text{rot}}_{ihm} = R_{imk} X_{iu} W^{\text{Q}}_{uhk}$ is practically free, leaving:

$$S^{\text{coupled}}_{ihj} = Q^{\text{rot}}_{ihm}\;\; \color{#ff4d4f}{R_{jmk'}}\;\; W^{\text{UK}}_{chk'}\;\; C_{jc}$$

- **If $\color{#ff4d4f}{R_{jmk'}}$ were absent**: we could absorb $W^{\text{UK}}$ *online* onto Query: $\tilde{Q}_{ihc} = Q^{\text{rot}}_{ihm} W^{\text{UK}}_{chm}$ (shape $(1, H, D_c)$, computed **once**), still caching only $C_{jc}$. Thus **$R_i$ is never the blocker!**
- **With $\color{#ff4d4f}{R_{jmk'}}$ present**: $R_{jmk'}$ (carrying $100\text{k}$ values of $j$) is locked via $k'$ onto $W^{\text{UK}}_{chk'}$ (carrying $H$ values of $h$):
  - **Push $W^{\text{UK}}$ left (onto Query)**: $\tilde{Q}_{i\color{#ff4d4f}{j}hc} = Q^{\text{rot}}_{ihm} R_{\color{#ff4d4f}{j}mk'} W^{\text{UK}}_{chk'}$ catches **$j$** $\implies$ shape explodes from $(1, H, D_c)$ to $(1, 100\text{k}, H, D_c)$!
  - **Keep $W^{\text{UK}}$ right (on Key)**: $K^{\text{rot}}_{j\color{#ff4d4f}{h}m} = R_{jmk'} W^{\text{UK}}_{c\color{#ff4d4f}{h}k'} C_{jc}$ keeps **$h$** $\implies$ caching it blows up HBM by $H\times$ (or recomputing it blows up FLOPs)!

### MLA's Solution (Decoupled RoPE): Split the Conflicting Axes

Split the dot product into two orthogonal branches ($q \cdot k = q^{\text{nope}}\cdot k^{\text{nope}} + q^{\text{rope}}\cdot k^{\text{rope}}$):
- **`nope` branch ($k$)**: keeps **$h$**, drops **$R_j$** $\implies$ full absorption, caches only $C_{jc}$ (**no $h$**).
- **`rope` branch ($r \to m$)**: keeps **$R_j$**, drops **$h$** from $K^{\text{rope}}$ $\implies$ tiny shared position key $K^{\text{rope}}_{jm}$ (**no $h$**):

$$Q^{\text{rope}}_{ihm} = R_{imr}\big(X_{iu}\, W^{\text{QR}}_{uhr}\big), \qquad K^{\text{rope}}_{jm} = R_{jmr'}\big(X_{ju}\, W^{\text{KR}}_{ur'}\big)$$

$$S_{ihj} = \frac{1}{\sqrt{D_k^{\text{nope}} + D_k^{\text{rope}}}}\Big(\underbrace{\tilde{Q}_{ihc}\, C_{jc}}_{\text{absorbed (no } h \text{ in } C_{jc}\text{)}} + \underbrace{Q^{\text{rope}}_{ihm}\, K^{\text{rope}}_{jm}}_{\text{shared } K^{\text{rope}}_{jm} \text{ (no } h\text{)}}\Big)$$

> **Index Sanity Check — Why Dropping $h$ on $K^{\text{rope}}_{jm}$ Doesn't Kill Quality:**
> 1. In $S^{\text{rope}}_{i\color{#1677ff}{h}j} = Q^{\text{rope}}_{i\color{#1677ff}{h}m} K^{\text{rope}}_{jm}$, **$h$ is still free on the LHS** via per-head $Q^{\text{rope}}_{ihm}$—every head still produces its own distinct distance pattern from one shared position tag, while semantic $K^{\text{nope}}_{jhk}$ and $V_{jhd}$ retain full per-head $h$.
> 2. Inside Softmax, orthogonal addition becomes a product: $P_{ihj} \propto \exp(S^{\text{nope}}_{ihj}) \times \exp(S^{\text{rope}}_{ihj})$, separating translation-invariant **content ("What")** from relative **distance ("Where")**.

---

## Showdown 4: Hardware & Memory Map (One Formula, Two Parenthesizations)

### 4A. KV Cache Footprint (DeepSeek-V2: $H=128,\; D_k^{\text{nope}}=D_v=128,\; D_c=512,\; D_k^{\text{rope}}=64$)

| Tensor | Indices | Scalars / Token / Layer | Hardware Role |
| :--- | :--- | :---: | :--- |
| Standard MHA: $K_{j\color{#ff4d4f}{h}k},\, V_{j\color{#ff4d4f}{h}d}$ | $j, \color{#ff4d4f}{h}, k/d$ | $2 \times 128 \times 128 = \mathbf{32{,}768}$ | **Eliminated** at decode (never materialized) |
| MLA Cache: $C_{jc}$ | $j, c$ (**no $h$**) | $\mathbf{512}$ | **HBM KV Cache**: shared across all $H$ heads |
| MLA Cache: $K^{\text{rope}}_{jm}$ | $j, m$ (**no $h$**) | $\mathbf{64}$ | **HBM KV Cache**: shared position tag across all $H$ heads |
| **Total MLA KV Cache** | **No $h$ anywhere** | **$576$ (98.2% smaller, ~57×)** | **SRAM reuse**: all $128$ heads share one $C_{jc}$ tile |

### 4B. Prefill vs. Decode = Two Contraction Orders of the Exact Same Expression

| Stage | Bottleneck | Bracketing $\to$ Attention step | Why it wins |
| :--- | :--- | :--- | :--- |
| **Prefill** (large $L_q$) | **Compute** | $\big(X_{iu} W^{\text{Q,nope}}_{uhk}\big)\big(C_{jc} W^{\text{UK}}_{chk}\big) \;\to\; Q_{ihk}\, K_{jhk}$ | **4× less attention math**: dot products sum over $k = 128$ instead of $c = 512$ ($K_{jhk}$ is discarded right after). |
| **Decode** ($L_q = 1$) | **Memory** | $\Big(X_{iu} \big(W^{\text{Q,nope}}_{uhk} W^{\text{UK}}_{chk}\big)\Big) C_{jc} \;\to\; \tilde{Q}_{ihc}\, C_{jc}$ | **57× less memory read**: load each $C_{jc}$ once and reuse it across all 128 heads (4× more math over $c = 512$, but decode waits on memory, not compute). |

---

## Bonus (Calculus Tie-In): Training Backprop Through the Shared Latent $C_{jc}$

| Traditional Matrix Calculus | Einstein Notation Tensor Calculus |
| :--- | :--- |
| Must unpack gradients from $H$ separate head slices, transpose each $(W^{\text{UK}}_{(h)})^\top$ and $(W^{\text{UV}}_{(h)})^\top$, and explicitly sum over $h$: | Apply the adjoint rule directly—since **$h$ is absent from LHS $\bar{C}_{jc}$**, summing across all $H$ heads falls out automatically: |
| $$\bar{C} = \sum_{h=1}^{H} \bar{K}^{(h)} \big(W^{\text{UK}}_{(h)}\big)^\top + \sum_{h=1}^{H} \bar{V}^{(h)} \big(W^{\text{UV}}_{(h)}\big)^\top$$ | $$\bar{C}_{jc} = \bar{K}^{\text{nope}}_{j\color{#fa8c16}{h}k}\, W^{\text{UK}}_{c\color{#fa8c16}{h}k} + \bar{V}_{j\color{#fa8c16}{h}d}\, W^{\text{UV}}_{c\color{#fa8c16}{h}d}$$ |
