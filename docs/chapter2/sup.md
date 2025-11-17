## 🧩 Part 1：Decoder-only 架构简介

我们今天讨论的大多数生成式 LLM 都属于 **Decoder-only** 架构。与原始 Transformer 模型包含独立的编码器（Encoder）和解码器（Decoder）不同，这类模型仅使用 Transformer 的解码器模块堆叠而成。

其核心组件包括：

1.  **因果自注意力 (Causal Self-Attention)**：这是架构的灵魂所在。在计算每个 token 的表示时，它只能“看到”自身及之前的所有 token，而不能“看到”未来的 token。这是通过一个“因果掩码”（Causal Mask）实现的，确保了模型生成文本时遵循从左到右的自然顺序。
2.  **前馈神经网络 (Feed-Forward Network)**：每个注意力层之后都会连接一个前馈网络，用于进行非线性变换和更深层次的特征提取。
3.  **层归一化 (Layer Normalization) 和残差连接 (Residual Connections)**：这些组件贯穿整个模型，是稳定深度网络训练过程的关键。

简单来说，Decoder-only 模型的核心任务就是：**根据已经输入的 token 序列，预测下一个最可能的 token 是什么**。

## 🚀 Part 2：训练 vs. 推理：两种截然不同的模式

模型的生命周期分为两个主要阶段：训练和推理。它们的计算模式和最终目标完全不同。

### 训练 (Training)：并行计算，学习模式

在训练阶段，我们的目标是让模型通过学习海量数据来掌握预测下一个词的能力。为此，我们使用一种叫做 **Teacher Forcing** 的机制。

*   **模式**：**并行计算**。
*   输入：我们将完整的文本序列（例如，“我是一个学生”）**一次性**全部输入模型。
*   计算过程：
    *   模型会并行地计算出序列中**每一个位置**的查询（$Q$）、键（$K$）和值（$V$）矩阵。
    *   注意力矩阵的计算也是一次性完成的，其维度为 (序列长度, 序列长度)。
    *   模型会并行地得到所有位置的预测输出，并与真实标签计算损失（Loss），然后通过反向传播来更新模型权重。
*   是否需要 KV Cache？
    *   完全不需要。因为在单次前向传播中，所有 token 的 $K$ 和 $V$ 都被同时计算并用于注意力计算，不存在“过去”和“未来”的区别，自然也就不需要缓存任何中间结果。

### 推理 (Inference)：串行计算，生成模式

在推理阶段，我们的目标是利用训练好的模型生成新的文本。这个过程是**自回归（Autoregressive）**的。

*   **模式**：**串行/序列计算**。
*   输入：我们先输入一个提示词（Prompt），然后模型生成一个 token；再将这个新 token 添加到输入的末尾，去生成下一个 token，如此循环往复。
*   计算挑战：如果每生成一个新 token，都把**至今为止的所有 token**（Prompt + 已生成的）重新完整地计算一遍，那么计算量将是巨大的。这引出了推理优化的必要性，也就是为什么推理过程需要被拆分为两个阶段。

## ⚙️ Part 3：推理的两个阶段：Prefill 与 Decoding

为了解决自回归生成中的重复计算问题，业界将推理过程优化为两个核心阶段：**Prefill（预填充）** 和 **Decoding（解码）**。

### 阶段一：Prefill (预填充)

这是处理用户输入（Prompt）的阶段，它是一个**并行计算**的过程，但**只执行一次**。

*   作用：
    1.  一次性处理整个 Prompt：对输入的 $L$ 个 token 进行一次完整的 Transformer 前向计算。
    2.  生成第一个输出 Token 的基础：计算出最后一个输入 token 的隐状态（hidden state），它将作为解码阶段生成第一个新 token 的起点。
    3.  初始化 KV Cache：这是 Prefill 阶段最关键的任务。它会计算出输入 Prompt 中每一个 token 在每一层的键（$K$）和值（$V$），并将它们存储起来。这个存储区就是 **KV Cache**。

*   计算复杂度：$O(L^2)$，其中 $L$ 是输入 Prompt 的长度。因为需要计算一个 $L \times L$ 的注意力矩阵。

> 第一步生成输出前，你必须先 prefill，因为要“看完”整个输入的意思。
>
> 根据对整个输入的prompt的理解，prefill阶段生成第一个回复的token，之后的每一步依次生成一个token。

### 阶段二：Decoding (解码)

当 Prefill 完成后，模型就进入了逐字生成的解码阶段。这是一个**串行计算**的过程。

*   作用：
    1.  高效生成后续 Token：每一步只处理一个新生成的 token。
    2.  利用并更新 KV Cache：在每一步生成中，无需重新计算整个序列的 $K$ 和 $V$。

*   **计算过程 (以生成第 $L+1$ 个 token 为例)**：
    1.  模型只接收上一步的输出作为输入，计算出新的查询向量 $q_{L+1}$。
    2.  从 KV Cache 中**读取**之前存储的所有 $K$ 和 $V$（即 $k_1, \dots, k_L$ 和 $v_1, \dots, v_L$）。
    3.  用新的 $q_{L+1}$ 与缓存中所有的 $K$ 进行注意力计算。
    4.  生成新的 token。
    5.  为这个新生成的 token 计算出它自己的 $k_{L+1}$ 和 $v_{L+1}$，并将它们**追加**到 KV Cache 中，供下一步使用。

*   **计算复杂度**：每一步的复杂度约为 $O(L_{\text{total}})$，其中 $L_{\text{total}}$ 是当前序列的总长度（Prompt + 已生成）。这是线性的，远快于 $O(L_{\text{total}}^2)$。

| 阶段         | 计算复杂度            | 输入长度          | 注意力矩阵大小              | 计算特点                   |
| :----------- | :-------------------- | :---------------- | :-------------------------- | :------------------------- |
| **Prefill**  | $O(L^2)$              | 整个 Prompt ($L$) | $L \times L$                | 并行计算，为后续生成做准备 |
| **Decoding** | $O(L_{\text{total}})$ | 当前 Token (1)    | $1 \times L_{\text{total}}$ | 串行计算，利用缓存，效率高 |

## 🧮 Part 4：核心功臣——KV Cache 详解

现在我们来聚焦这个实现高效推理的最大功臣：**KV Cache**。

### 1. KV Cache 是什么？

KV Cache 是一个用于存储 Transformer 注意力层中**键（$K$）**和**值（$V$）**矩阵的内存区域。在自回归生成过程中，一旦某个 token 的 $K$ 和 $V$ 被计算出来，它们在后续的生成步骤中是**固定不变的**。因此，将它们缓存起来可以避免大量的重复计算。

### 2. KV Cache 如何生效？

![img](./pictures/sup/1*uyuyOW1VBqmF5Gtv225XHQ.gif)

如图所示，在没有 KV Cache 的情况下，当要生成第 $i$ 个 token 时，模型不仅需要计算当前 token 的注意力，还必须重复计算第 $1$ 到 $i-1$ 个 token 之间已经算过的所有注意力关系。

有了 KV Cache 后，每一步我们只需要计算当前 token 的 $Q, K, V$。在计算注意力分数时，也只需要计算出注意力矩阵的第 $i$ 行，即当前 token 与之前所有 token ($1 \to i$) 之间的注意力分数即可。

我们通过复杂度分析来更具体地看一下：

**当不使用 KV Cache 时：**
在生成第 $i$ 个 token 时，你需要将**整个长度为 $i$ 的序列（从 token 1 到 $i$）**全部重新传入模型。

1.  **计算**：
    *   **线性变换 ($Q, K, V$)**：输入 $X$ 的 `shape` 为 $(i, \text{dim})$。计算 $Q, K, V$ 需要进行 $(i, \text{dim}) \times (\text{dim}, \text{dim})$ 的矩阵乘法，复杂度为 $O(i \cdot \text{dim}^2)$。因为要重新计算从 $1$ 到 $i$ 所有 token 的 $K$ 和 $V$。
    *   **注意力计算 ($Q K^T$)**：$Q$ 的 `shape` 为 $(i, \text{dim})$，$K^T$ 的 `shape` 为 $(\text{dim}, i)$。计算 $Q K^T$ 的复杂度为 $O(i^2 \cdot \text{dim})$。
    *   **值加权 (Attn $\cdot$ V)**：Attn 的 `shape` 为 $(i, i)$，$V$ 的 `shape` 为 $(i, \text{dim})$。计算 Attn $\cdot V$ 的复杂度为 $O(i^2 \cdot \text{dim})$。

2.  **总和** (生成 $T$ 个 token)：
    *   **线性变换总和**：$\sum_{i=1}^T O(i \cdot \text{dim}^2) = O(\text{dim}^2) \cdot \sum_{i=1}^T i = O(T^2 \cdot \text{dim}^2)$。
    *   **注意力总和**：$\sum_{i=1}^T O(i^2 \cdot \text{dim}) = O(\text{dim}) \cdot \sum_{i=1}^T i^2 = O(T^3 \cdot \text{dim})$。

总复杂度为 $O(T^2 \cdot \text{dim}^2 + T^3 \cdot \text{dim})$，可以看到它与序列长度 $T$ 的**三次方**成正比。

**当使用 KV Cache 时：**
在生成第 $i$ 个 token 时，我们只向模型输入**第 $i$ 个 token**（`shape` 为 $(1, \text{dim})$）。而前 $i-1$ 个步骤的 $K$ 和 $V$ 已存储在缓存中。

1.  **计算**：
    *   **线性变换 ($q, k, v$)**：计算**新**的 $q_i, k_i, v_i$。输入 `shape` 为 $(1, \text{dim})$，复杂度为 $O(1 \cdot \text{dim}^2) = O(\text{dim}^2)$。我们只需要计算当前 token 的 $Q, K, V$。
    *   **合并缓存**：将 $k_i, v_i$ 添加到 $K_{\text{cache}}, V_{\text{cache}}$。
        *   $Q = q_i$ (`shape`: $1, \text{dim}$)
        *   $K_{\text{cache}}$ (`shape`: $i, \text{dim}$)
        *   $V_{\text{cache}}$ (`shape`: $i, \text{dim}$)
    *   **注意力计算 ($q K^T$)**： $(1, \text{dim}) \times (\text{dim}, i)$，复杂度为 **$O(i \cdot \text{dim})$**。
    *   **值加权 (Attn $\cdot$ V)**： $(1, i) \times (i, \text{dim})$，复杂度为 **$O(i \cdot \text{dim})$**。
2.  **总和** (生成 $T$ 个 token)：
    *   **线性变换总和**：$\sum_{i=1}^T O(\text{dim}^2) = O(T \cdot \text{dim}^2)$。
    *   **注意力总和**：$\sum_{i=1}^T O(i \cdot \text{dim}) = O(\text{dim}) \cdot \sum_{i=1}^T i = O(T^2 \cdot \text{dim})$。

总复杂度为 $O(T \cdot \text{dim}^2 + T^2 \cdot \text{dim})$，与序列长度 $T$ 的**平方**成正比。

从 $O(T^3)$ 到 $O(T^2)$ 的复杂度降低是革命性的，这使得长文本的流畅、高效生成在实践中成为可能。

| 复杂度维度                | 不使用 KV Cache (W/O KV Cache)                     | 使用 KV Cache (W/ KV Cache)                      |
| :------------------------ | :------------------------------------------------- | :----------------------------------------------- |
| **生成第 $i$ 步的复杂度** | $O(i \cdot \text{dim}^2 + i^2 \cdot \text{dim})$   | $O(\text{dim}^2 + i \cdot \text{dim})$           |
| **生成 $T$ 步的总复杂度** | $O(T^2 \cdot \text{dim}^2 + T^3 \cdot \text{dim})$ | $O(T \cdot \text{dim}^2 + T^2 \cdot \text{dim})$ |

### 3. Prefill 和 Decoding 都会保存 KV Cache 吗？

是的，但它们扮演着不同的角色：

*   **Prefill 阶段**：**创建并初始化** KV Cache。它一次性计算出输入 Prompt 中所有 token 的 $K$ 和 $V$，并将其填充到 Cache 中。
*   **Decoding 阶段**：**使用并扩展** KV Cache。在每一步，它会利用已有的 Cache 进行计算，并把自己新生成的 token 的 $K$ 和 $V$ 追加进去。

> Prefill 负责一次性填充缓存，Decoding 则负责逐步使用并扩展缓存。
