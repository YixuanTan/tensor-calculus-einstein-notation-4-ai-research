这里已将当前 NotebookLM 项目中沉淀的所有**学术定位、符号规范、板书设计系统、核心推导主线（Linear Attention & FlashAttention Backward）、Side-by-Side 对比范式及代码验证规范**整理为一份可以直接在 Antigravity 中使用的 Markdown 规范文档：

---

```markdown
# AGENTS.md / TENSOR_SPEC.md: AI Research 视频与张量推导工作流规范

本规范用于指导大模型数学基础与工业级算子推导的设计、LaTeX 讲义编写及配套代码验证，核心宗旨是**用严谨的爱因斯坦求和记号（Einstein Notation）降维打击二维矩阵混乱与 Transpose Hell**[cite: 1, 2]。

---

## 1. 核心定位与技术栈基底

- **作者身份生态位**：计算力学（Computational Mechanics）背景，将张量分析、指标缩并与连续介质力学思维迁移至大模型底层（SysML / 架构演进 / 数值收敛分析）[cite: 2]。
- **核心叙事模式（Proof of Human Work）**：
  - 杜绝 AI 生成水文感，以**“Wacom + Xournal++ 深色黑板手推 + 4~5 行高密度指标核心”**为载体[cite: 1, 2]。
  - **必选叙事范式**：必须包含 **Side-by-Side 对比**（左栏：传统矩阵与转置泥潭；右栏：指标记号与结合律秒杀）[cite: 2]。
- **硬件与软件工作流**：
  - 硬件：Wacom 数位板 + Mac Studio Display + Studio Display 麦克风与摄像头[cite: 2]。
  - 软件：Xournal++（深色网格背景，原生 LaTeX 公式渲染）+ OBS Studio（画中画录制）+ CapCut/剪映（自动识别英汉混杂专业字幕）[cite: 1, 2, 3]。

---

## 2. 张量指标与书写符号规范 (Tensor Index Conventions)

为避免指标重名与维度碰撞，所有数学推导严格遵循以下符号系统[cite: 1]：

### 2.1 轴名称与指标分配
| 维度类别 | 指标符号 | 物理含义与取值范围 | 规则与约定 |
| :--- | :--- | :--- | :--- |
| **Batch / Head** | $b, h$ | $b \in [1, B], h \in [1, H]$ | 在非 GQA 推导中作为**旁观自由指标（Spectator Free Indices）**，或直接设 $B=1, H=1$[cite: 1, 2]。 |
| **序列位置 (Sequence)** | $i$ | Query 序列位置 ($i \in [1, L]$) | 核心瓶颈轴（推导中用**亮黄色**高亮）[cite: 1, 3]。 |
| **序列位置 (Sequence)** | $j$ | Key/Value 序列位置 ($j \in [1, L]$) | 缩并求和哑指标（Dummy Index）[cite: 1]。 |
| **归一化位置 (Sequence)** | $m$ 或 $n$ | Softmax 分母求和位置 ($m, n \in [1, L]$) | **严禁与 $j$ 混用**，专门用于分母全序列归一化[cite: 1]。 |
| **特征通道 (Channel/Dim)** | $k, m$ | Head Dim / 特征隐层 ($k \in [1, D]$ 或 $M$) | 轻量轴（推导中用**青色/浅蓝色**高亮）[cite: 1, 3]。 |
| **输出通道 (Output Dim)** | $d$ | Value / Output 特征维 ($d \in [1, D_v]$) | 输出自由指标[cite: 1]。 |

### 2.2 记号书写与求和冻结原则
1. **分量即向量**：无需人为添加多余的帽子（$\hat{Q}$ 为单位向量/估计值）或横线（$\bar{Q}$ 为均值）[cite: 1]。直接用 $Q_{ik}$（$i$ 固定，$k$ 为自由分量）代表第 $i$ 个 Token 向量[cite: 1]。宏观示意时可选用 $\vec{Q}_i$ 或 $Q_i$[cite: 1]。
2. **重复指标不求和的括号冻结原则 (Parenthesized Non-Summation Rule) 与左侧自由指标免括法则**：
   - **左侧自由指标免括法则（LHS Free Index Invariance，消除冗余）**：
     - 若等式左侧（LHS）已经显式给出了该自由指标（如 $P_{ij} = \frac{\exp(S_{ij})}{L_i}$），根据张量分析自由指标守恒律，等号左侧的自由指标 $i$ 强行约束等号右侧的 $i$ 必须同为自由指标，**此时右侧的 $i$ 严禁加括号，加括号属于不必要的视觉冗余**（对应 PyTorch einsum `'ij,i->ij'`，左右两端轴一一对应，无歧义）[cite: 2]。
   - **何时必须使用圆括号冻结 $(\cdot)$**：
     - **孤立表达式 / 无 LHS 明确约束的单项**：当没有完整等式左侧约束，但同一项中出现了重复指标且**明确不进行缩并求和**时（如局部讨论 $D_{(i)} dP_{(i)j}$ 或行广播项 $\frac{\exp(S_{(i)j})}{L_{(i)}}$），必须加圆括号冻结，以防止被经典爱因斯坦约定误判为求和哑指标[cite: 2]。
   - **无歧义原则**：有左侧看左侧自由指标守恒；无左侧看单项括号冻结。确保推导既符合最严谨的连续介质力学规范，又保持数学公式的极简美感[cite: 2]。

---

## 3. 核心专题推导模板库

### 专题一：Linear Attention 的结合律与核函数解耦
- **核心痛点**：Softmax 为非线性指数，导致 $\exp(Q_{ik} K_{jk})$ 中 $i$ 与 $j$ 强耦合（求和号 $\sum_j$ 无法穿透 $\exp$），强行将括号锁死在左侧，必须实例化 $L \times L$ 矩阵[cite: 1, 3]。
- **核函数变量分离**：令 $\text{Sim}(Q_i, K_j) = \tilde{Q}_{ik} \tilde{K}_{jk}$（其中 $\tilde{Q} = \phi(Q) \ge 0$）[cite: 1, 3]。
- **括号调换推导**：
  $$\text{Numerator}_{id} = \sum_j (\tilde{Q}_{ik} \tilde{K}_{jk}) V_{jd} = \tilde{Q}_{ik} \underbrace{\left( \sum_j \tilde{K}_{jk} V_{jd} \right)}_{S_{kd} \in \mathbb{R}^{D \times D}}$$
[cite: 1, 3]
- **算力收益**：复杂度从 $\mathcal{O}(L^2 D)$ 跌至 $\mathcal{O}(L D^2)$[cite: 1, 3]。
- **自回归 RNN 等价性**：$S_{kd}^{(t)} = S_{kd}^{(t-1)} + \tilde{K}_{tk} V_{td}$（Rank-1 外积累加，常数显存）[cite: 1, 3]。
- **学术拔高与阿喀琉斯之踵**：固定大小的 $D \times D$ 状态存在秩瓶颈（Capacity/Rank Bottleneck），导致长程精确检索/大海捞针能力下降，从而引出现代 Frontier Lab 的 Hybrid（混合架构）演进[cite: 1, 3]。

### 专题二：FlashAttention 反向求导与 $D_i$ 消除证明
- **矩阵痛点**：$\text{diag}$、哈达玛积 $\odot$ 与多维张量转置符号失灵[cite: 1, 2]。
- **核心克罗内克导数**：
  $$\frac{\partial P_{im}}{\partial S_{ij}} = P_{im} (\delta_{jm} - P_{ij}) \implies dS_{ij} = P_{ij} \left( dP_{ij} - \sum_m P_{im} dP_{im} \right)$$
[cite: 2]
- **$D_i = O_{id} dO_{id}$ 秒杀证明**：
  $$\sum_m P_{im} dP_{im} = P_{im} (dO_{id} V_{md}) \overset{\text{结合律}}{=} (P_{im} V_{md}) dO_{id} = O_{id} dO_{id} \equiv D_i$$
[cite: 1, 2]
- **工程结论**：反向传播仅需缓存 $O(L)$ 的一阶向量 $D_i$，中间概率图 $P$ 全局显存开销彻底归零[cite: 1, 2]。

---

## 4. 标准 Side-by-Side 对比板书规范 (ASCII 排版)

```text
┌─────────────────────────────────────────┬─────────────────────────────────────────┐
│ ❌ Standard Matrix / PyTorch Notation   │ ✅ Einstein Notation (Tensor Indices)   │
├─────────────────────────────────────────┼─────────────────────────────────────────┤
│ 1. Forward / Conceptual:                │ 1. Forward Setup:                       │
│    A = Q @ K.T                          │    S_ij = (1/√d_k) Q_ik * K_jk          │
│    O = A @ V                            │    P_ij = softmax_j(S_ij),  O_id = P_ij * V_jd
│                                         │                                         │
│ 2. 4D Production Reality (B, H, L, D):  │ 2. 4D Production Reality:               │
│    Q.transpose(1, 2) @                  │    Q_bhik * K_bhjk                      │
│    K.transpose(1, 2).transpose(-2, -1)  │    (Axes stay explicitly named;         │
│    (Transpose Hell & contiguous bugs)   │     zero stride/permutation ambiguity)  │
│                                         │                                         │
│ 3. Associative Law / Linear Attention:  │ 3. Associative Law (Linear Attention):  │
│    O = Q @ (K.T @ V) ???                │    O_id = Q_ik * (K_jk * V_jd)          │
│    Which matrix needs transpose?        │    Contract sequence axis j first!      │
│    Dimension alignments get messy!      │    Collapses into state S_kd in O(L).   │
│                                         │                                         │
│ 4. Backward Pass (The Flash Trick):     │ 4. Backward Pass (The Flash Trick):     │
│    D = rowsum(dP ⊙ P)                   │    D_i = P_im * dP_im                   │
│    Requires complex matrix traces to    │        = P_im * (dO_id * V_md)          │
│    prove D = rowsum(dO ⊙ O).            │        = (P_im * V_md) * dO_id          │
│                                         │        = O_id * dO_id                   │
└─────────────────────────────────────────┴─────────────────────────────────────────┘

```



---

## 5. 配套 PyTorch Einsum 单元测试标准

每个推导必须配套可复现的 Python 验证脚本（10~15 行），采用 `torch.einsum` 进行数值一致性断言：

```python
import torch

def verify_linear_attention_associative_law():
    B, H, L, D = 2, 4, 1024, 64
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)

    # 模拟非负特征映射
    phi_Q = torch.nn.functional.elu(Q) + 1.0
    phi_K = torch.nn.functional.elu(K) + 1.0

    # 路径 A：传统左结合 (生成 L x L 注意力图，O(L^2))
    A = torch.einsum('bhik,bhjk->bhij', phi_Q, phi_K)
    O_quad = torch.einsum('bhij,bhjd->bhid', A, V)

    # 路径 B：结合律右结合 (沿序列轴 j 缩并，生成 D x D 隐状态，O(L))
    S = torch.einsum('bhjk,bhjd->bhkd', phi_K, V)
    O_linear = torch.einsum('bhik,bhkd->bhid', phi_Q, S)

    assert torch.allclose(O_quad, O_linear, atol=1e-4)
    print("✓ Linear Attention Associative Law Numerically Verified!")

def verify_flash_backward_identity():
    B, L, D = 1, 64, 32
    Q = torch.randn(B, L, D, requires_grad=True)
    K = torch.randn(B, L, D, requires_grad=True)
    V = torch.randn(B, L, D, requires_grad=True)

    S = torch.einsum('bik,bjk->bij', Q, K) / (D ** 0.5)
    P = torch.softmax(S, dim=-1)
    O = torch.einsum('bij,bjd->bid', P, V)
    dO = torch.randn_like(O)

    # 标准中间梯度与 FlashAttention 标量化对比
    dP = torch.einsum('bid,bjd->bij', dO, V)
    D_standard = (P * dP).sum(dim=-1, keepdim=True)
    D_flash = (O * dO).sum(dim=-1, keepdim=True)

    assert torch.allclose(D_standard, D_flash, atol=1e-5)
    print("✓ FlashAttention D_i = sum(O * dO) Identity Verified!")

if __name__ == "__main__":
    verify_linear_attention_associative_law()
    verify_flash_backward_identity()

```



---

## 6. 社媒发布与学术资产沉淀规范

* **LinkedIn / X (Twitter)**：
* 配套沉淀为 2~3 页精美 LaTeX PDF 讲义（Document Post 形式具备翻页轮播优势）。


* 动态文案聚焦痛点词汇：*“without getting lost in transpose hell”*、*“simply swapping the contraction brackets drops complexity from $O(L^2)$ to $O(L)$”*。




* **B 站 / 国内平台**：
* 核心标签：大模型、Transformer、张量分析、LinearAttention、爱因斯坦求和、PyTorch。


* 评论区置顶 10 行验证代码及推导核心对比，引导学术探讨。





```

---

### 如何在 Antigravity 中生效：
1. 在你的项目根目录下新建一个名为 `AGENT_INSTRUCTIONS.md` 或 `TENSOR_SPEC.md`。
2. 将上述内容直接粘贴进去。
3. 后续在 Antigravity 中提问、编写 LaTeX 讲义代码或写算子测试时，直接在对话中输入 `@TENSOR_SPEC.md`，Agent 就会严格遵循这套指标记号、配色规范与推导逻辑[cite: 1, 3]。

<FollowUp label="需要我为你生成一份配套该规范的 LaTeX 讲义模板代码（tensor_template.tex）吗？" query="请根据上述张量规范为我生成一份可以直接编译为 2 页 PDF 讲义的 LaTeX 模板代码 (tensor_template.tex)。"/>

```