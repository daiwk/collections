---
title: "LLM 网络结构发展趋势：从 2023 年到未来"
author: ""
date: ""
lang: zh-CN
---

> Pandoc 转 PDF 建议使用 `-f markdown+tex_math_dollars+smart`，这样 `$...$` 行内公式会更稳。

# 一句话结论

2023 至今，LLM 网络结构的主线不是 **Transformer 被某个新结构彻底取代**，而是：

> Transformer 被拆成若干可替换部件，然后分别围绕容量、长上下文、推理吞吐和部署成本做系统性重构。

传统 decoder-only Transformer block 可以粗略写成：$x → Token Mixer → Channel Mixer → Residual / Norm$。

其中：

- **Token Mixer**：负责 token 间信息交互，典型代表是 self-attention。
- **Channel Mixer**：负责每个 token 内部 hidden 维度变换，典型代表是 FFN / MLP。
- **Residual / Norm**：负责稳定训练和信号传播。

现在的演变方向大致是：

Token mixer 的演变可以概括为：Self-Attention $→$ GQA / MLA $→$ GDN / KDA / Mamba-like $→$ CSA / HCA / Sparse / Compressed Attention。

Channel mixer 的演变可以概括为：Dense FFN $→$ MoE FFN。

Memory / KV 的演变可以概括为：Full KV Cache $→$ Latent KV / Compressed KV $→$ Recurrent State / Hybrid State。

所以未来 LLM 更像：

> 稀疏专家容量池 + 混合 token mixer + 压缩记忆系统 + 面向长上下文和高吞吐推理共同设计的执行图。

# 先澄清几个术语

## Token mixer 是什么？

这里的 **token mixer** 不是推荐系统里的 RankMixer 模块名，而是一个更通用的结构抽象。

它指的是：

> 负责让序列中不同 token 之间交换信息的模块。

在不同模型里，token mixer 可以是：

| 模型/路线 | Token mixer |
|---|---|
| 标准 Transformer | Self-Attention |
| Mistral 类模型 | Sliding Window Attention + GQA |
| DeepSeek-V2/V3 | MLA |
| Qwen3-Next | Gated DeltaNet + Gated Attention |
| Kimi Linear | KDA + MLA |
| Hunyuan TurboS/T1 | Mamba2 + Attention |
| DeepSeek V4 | CSA + HCA |

它本质上解决的问题是：

> 第 $i$ 个 token 如何感知、融合、压缩、检索其他 token 的信息？

## Channel mixer 是什么？

**Channel mixer** 指的是对每个 token 内部 hidden 维度做变换的模块。

标准 Transformer 里就是 FFN / MLP：

标准 Transformer 里可以写成：`x[:, t, :] -> FFN -> y[:, t, :]`。

它通常不直接混不同 token 位置，而是对每个位置独立做非线性变换。

近几年 channel mixer 的主要变化是：

Channel mixer 的演变可以概括为：Dense FFN $→$ MoE FFN。

也就是说，FFN 不再是所有 token 都走同一个 dense 网络，而是每个 token 被路由到少数几个 expert。

## MoE 优化的是什么？

MoE 主要优化的是：

> 模型容量如何扩大，同时每个 token 的计算量不同比例扩大。

它的核心指标有两个：

- $P_{\text{total}}$：模型总参数量。
- $P_{\text{active}}$：每个 token 实际激活的参数量。

对于 dense 模型，通常有：

对于 dense 模型，通常有 $P_{\text{total}} \approx P_{\text{active}}$。

对于 MoE 模型，则是：

对于 MoE 模型，则是 $P_{\text{total}} \gg P_{\text{active}}$。

所以 MoE 的本质是：

> 用稀疏激活换更大的知识容量和模型容量。

它不直接解决 token 间交互成本，也不直接替代 attention。

## KV cache 为什么变成核心瓶颈？

LLM 推理分两个阶段：

- **Prefill**：输入 prompt，计算所有输入 token 的 hidden states 和 KV。
- **Decode**：每次生成一个新 token，并读取历史 KV cache。

在长上下文、长输出、多轮 agent 场景下，decode 阶段经常被 KV cache 卡住：

- KV cache 占显存。
- 每步 decode 都要读历史状态。
- 长上下文下 memory bandwidth 压力很大。
- batch 变大时 cache 管理更复杂。

KV cache 的量级可以粗略写成：

KV cache 的量级可以粗略写成：$\mathrm{KVCache} \propto L \times H_{\text{kv}} \times d_{\text{head}} \times N_{\text{layer}}$。

其中 L 是上下文长度，H_kv 是 KV head 数量，d_head 是 head dimension，N_layer 是层数。

因此很多新架构实际上都在解决：

> 每生成一个 token，需要读取多少历史状态？这些状态能不能被压缩？能不能不用每层都保留完整 KV？

# 2023：优化标准 Transformer

## 这一阶段的主流结构

2023 年主流 LLM 仍然是 decoder-only Transformer：

2023 年主流 LLM 仍然是 decoder-only Transformer：Embedding $\rightarrow [\mathrm{SelfAttention}+\mathrm{FFN}]^N \rightarrow$ LM Head。

这一阶段的关键不是重写整个 block，而是在标准 Transformer 上做增量优化。

## 典型优化：MHA 变成 MQA / GQA

标准 multi-head attention 里，每个 query head 通常都有自己的 key/value head。

问题是：

问题是：$H_{\text{kv}}$ 越大，KV cache 越大。

于是出现：

- **MQA**：多个 query head 共享一组 KV head。
- **GQA**：多个 query head 分组共享 KV head。

这类优化的本质是：

> 减少 KV head 数量，从而减少 decode 阶段的 KV cache 读写成本。

## 典型优化：Sliding Window Attention

全量 causal attention 的复杂度会随上下文增长而上升。Sliding Window Attention 的做法是：

Sliding Window Attention 可以写成：$\mathrm{Attn}(x_t)=\mathrm{Attn}(x_{t-w}, \ldots, x_t)$，其中 $w$ 是窗口大小。

其中 $w$ 是窗口大小。

优点是降低长上下文计算量，更适合局部依赖强的场景。缺点是远距离精确依赖能力受限，需要和 global attention 或其他机制结合，才能处理真正全局的信息。

## 基础配置逐渐稳定

这一阶段还形成了一些基础配置共识：

- **RoPE**：主流位置编码方案。
- **RMSNorm**：更轻量的 norm。
- **SwiGLU**：更强的 FFN 激活结构。
- **Pre-Norm**：更稳定的深层训练结构。

这些不是特别激进的架构替换，但共同构成了现代 LLM block 的基础配方。

## 2023 总结

2023 的核心特征是：

- 主干仍是 Transformer。
- Attention 仍是每层核心 token mixer。
- 优化重点是让 attention 更便宜、更适合推理。

一句话：

> 2023 是“优化 Transformer”的阶段。

# 2023 末到 2024：MoE 化 Transformer

## 为什么 MoE 成为主线？

大模型能力提升通常需要更大的容量。

但 dense 模型有一个问题：

但 dense 模型有一个问题：$P_{\text{total}} \uparrow \Rightarrow \mathrm{FLOPs/token} \uparrow$。

MoE 的思路是：

MoE 的思路是：$P_{\text{total}} \uparrow$，同时让 $P_{\text{active}}$ 相对可控。

这使得模型可以同时获得更大的总容量和相对可控的每 token FLOPs。

## MoE 改的是 FFN，不是 attention

标准 Transformer block 可以写成：

标准 Transformer block 可以写成：Attention $→$ Dense FFN。

MoE Transformer block 则变成：

MoE Transformer block 则变成：Attention $→$ MoE FFN。

所以 MoE 主要替换的是 channel mixer，而不是 token mixer。

它不是在问：

> token 和 token 之间还要不要 attention？

而是在问：

> 每个 token 的 FFN 计算，是否可以只走少数专家？

## MoE 的核心挑战

### Expert routing

每个 token 要选择哪些 expert。常见方式是 top-$k$ routing：

每个 token 要选择哪些 expert，常见方式是 top-$k$ routing：$\mathrm{Experts}(x_t)=\mathrm{TopK}(\mathrm{Router}(x_t), k)$。

核心问题包括：

- routing 是否稳定？
- expert 是否负载均衡？
- token 是否会集中到少数 expert？

### Load balancing

如果 expert 负载不均衡，会导致：

- 某些 expert 过载。
- 通信成本上升。
- 训练效率下降。
- 推理延迟不稳定。

因此 MoE 通常需要负载均衡损失或其他 balancing 策略。

### Expert parallel

MoE 模型往往需要 expert parallel：

在 expert parallel 中，不同 expert 会分布在不同 GPU / 节点上。

这会引入 all-to-all communication。因此 MoE 不只是模型结构问题，也是分布式系统问题。

## 这一阶段的结构范式

可以抽象成：

这一阶段的结构可以抽象成：Embedding $→ [Attention+MoE\ FFN]^N →$ LM Head。

相比 2023：

- Token mixer 变化不大。
- Channel mixer 从 Dense FFN 变成 MoE FFN。

## 2024 总结

这一阶段的核心特征是：

- 大模型开始从 dense Transformer 转向 sparse MoE Transformer。
- 总参数增长很快，但 active params 被控制。
- MoE 成为扩大模型容量的主线。

一句话：

> 2024 是“MoE 化 Transformer”的阶段。

# 2024 到 2025：KV / Memory 优化成为核心问题

## 为什么 KV cache 成为瓶颈？

MoE 解决了 FFN 侧容量扩展问题，但 attention 侧的问题还在。

尤其是长上下文和长输出场景下：

尤其是长上下文和长输出场景下，每生成一个 token，都要读取历史 KV cache。

如果上下文很长，KV cache 会带来显存占用、memory bandwidth、batch size 和多用户 serving 的压力。

这时推理瓶颈往往不是纯 FLOPs，而是：

这时推理瓶颈往往不是纯 FLOPs，而是 memory bandwidth 和 KV cache access。

## GQA / MQA 的局限

GQA 和 MQA 能减少 KV head 数量，但它们仍然保留完整 token 维度的 KV cache：

KV cache 的量级可以粗略写成：$\mathrm{KVCache} \propto L \times H_{\text{kv}} \times d_{\text{head}} \times N_{\text{layer}}$。

当上下文从 32K、128K 走向 1M 时，仅仅减少 KV heads 可能不够。

## MLA 的意义

MLA 可以理解为更进一步的 KV 压缩路线。

核心思想是：

MLA 的核心思想可以理解为：$(K,V) \rightarrow z_{\text{latent}}$。

也就是不要直接缓存完整 K/V，而是缓存压缩后的 latent 表示。

这样做的目标是：

- 降低 KV cache 显存。
- 降低 decode 读取历史状态的带宽。
- 保留较强 attention 表达能力。

## Memory 设计从工程细节变成架构中心

过去大家聊模型结构，重点是：

- 多少层？
- hidden size 多大？
- head 数多少？
- FFN 多大？

现在越来越要问：

- 每层是否需要 KV cache？
- KV cache 有多大？
- 能否压缩？
- 能否复用？
- 能否只保留部分全局状态？

这说明 memory/KV 已经从推理工程细节变成网络结构设计中心。

## 这一阶段总结

这一阶段的核心特征是：

> Attention 没有被替代，但 attention 的 memory 形式被重构。

一句话：

> 2024 到 2025 是“KV / Memory 优化成为架构核心”的阶段。

# 2025：Hybrid Token Mixer 进入主线

## 从“优化 attention”到“减少 full attention”

到了 Qwen3-Next、Kimi Linear、Hunyuan TurboS/T1 这类模型，一个更激进的问题出现了：

> 每一层都需要 full attention 吗？

如果每层都是 full attention：

- 长上下文 prefill 很贵。
- decode KV cache 很大。
- 1M context 很难高效 serving。

于是新的方向是：

> 大部分层使用更便宜的 token mixer，少数层保留 global attention / MLA。

## Qwen3-Next 路线：Gated DeltaNet + Attention

Qwen3-Next 的典型思路是：

Qwen3-Next 的结构可以抽象为：$[\mathrm{GDN}+\mathrm{MoE}]^3 \rightarrow [\mathrm{Attention}+\mathrm{MoE}]$。

这种结构的关键不是“彻底抛弃 attention”，而是：

> 让大多数层不再承担 full attention 的成本，只在周期性关键层做全局交互。

## Kimi Linear 路线：KDA + MLA

Kimi Linear 类似，也不是纯 linear。

它的思路可以概括为：

Kimi Linear 的结构可以概括为：$[\mathrm{KDA}+\mathrm{MoE}]^3 \rightarrow [\mathrm{MLA}+\mathrm{MoE}]$。

KDA 更偏 recurrent / finite-state memory，MLA 负责保留全局 attention 能力。

这说明 linear 路线的主流设计也不是所有 attention 全部删除，而是：

> 多数层便宜状态更新，少数层全局校准。

## Hunyuan TurboS/T1 路线：Mamba + Attention + MoE

混元之前确实有 hybrid Transformer-Mamba-MoE 路线。

其关键思想也是：

混元 TurboS/T1 的关键组合可以概括为：Mamba/SSM + Attention + MoE。

其中 Mamba / SSM 负责长序列低成本状态传播，attention 负责关键全局交互，MoE 负责容量扩展。

这说明 Mamba / SSM 并不是被完全否定，而是作为 hybrid token mixer 的一种选择。

## 为什么不是全换成 Linear / Mamba？

因为 full/global attention 仍然有重要价值：

- 精确 copy。
- 长距离检索。
- 复杂 in-context learning。
- 工具调用轨迹对齐。
- 多文档交叉引用。
- 复杂 reasoning 中的全局信息聚合。

Linear / recurrent / Mamba-like 结构虽然省 cache、省带宽，但可能在精确检索和全局对齐上需要 attention 兜底。

因此更合理的方向是：

因此更合理的方向不是 **Attention vs. Linear**。

而是：

更合理的方向是 **Attention + Linear/Recurrent Mixer**。

## 这一阶段总结

2025 的核心结构变化是：

> 每层 full attention 不再是默认答案，hybrid token mixer 成为主线探索方向。

一句话：

> 2025 是“Hybrid Token Mixer + MoE”的阶段。

# 2026：DeepSeek V4 与 Compressed / Sparse Attention 路线

## DeepSeek V4 的重要信号

DeepSeek V4 体现了另一种思路：

> 不一定用 GDN / KDA / Mamba 来替代 attention，也可以继续保留 attention 语义，但对 attention 做压缩和稀疏化。

核心方向包括：

- Compressed Sparse Attention，简称 CSA。
- Heavily Compressed Attention，简称 HCA。
- 更极致的 KV / memory 压缩。
- 面向 1M context 的架构设计。

## Compressed / Sparse Attention 与 Linear 路线的区别

### Linear / Recurrent 路线

代表包括 GDN、KDA、Mamba-like。

核心思想是：

Linear / recurrent 路线的核心思想是：历史序列 $→$ recurrent state / finite state。

优点：

- 长 decode cache 小。
- 状态更新成本低。
- 长上下文吞吐潜力大。

风险：

- 精确检索能力需要验证。
- 需要周期性 attention 兜底。
- kernel / 训练稳定性要求高。

### Compressed / Sparse Attention 路线

代表包括 CSA、HCA、Sparse Attention、Compressed KV。

核心思想是：

Compressed / sparse attention 路线的核心思想是：保留 attention 访问形式，同时减少需要访问的 token / KV 表示。

优点：

- 更接近 attention 原始语义。
- 精确交互能力可能更稳。
- 更容易和现有 attention 生态衔接。

风险：

- sparse pattern 设计复杂。
- cache 管理复杂。
- kernel 也需要专门优化。

## 未来不会只有一个赢家

更可能出现三条路线并行：

| 路线 | 代表 | 优点 | 风险 |
|---|---|---|---|
| Hybrid Linear / Recurrent | Qwen3-Next, Kimi Linear | KV/state 小，长 decode 友好 | 精确检索需 attention 兜底 |
| Compressed / Sparse Attention | DeepSeek V4 | 保留 attention 语义，长上下文成本低 | sparse/cache/kernel 复杂 |
| Conservative Attention-MoE | Hy3-preview 等 | 生态成熟，工程风险低 | 1M context 成本压力大 |

所以未来很可能不是 A 打败 B，而是：

> 不同层、不同模型规模、不同场景采用不同 token mixer 组合。

# 从 2023 到未来的总时间线

可以把演变压缩成下面这条线：

总时间线可以概括为：2023 优化 Transformer，2024 MoE 化 Transformer，2025 混合化 token mixer，2026+ 面向长上下文和 serving 重构整个 block。

更细一点：

| 阶段 | 结构特征 | 主要目标 |
|---|---|---|
| 2023 | Dense Transformer + GQA / MQA / SWA | 让标准 attention 更便宜 |
| 2024 | Sparse MoE Transformer | 扩大 total params，控制 active params |
| 2024--2025 | MLA / KV Compression | 降低 KV cache 与 decode memory bandwidth |
| 2025 | Hybrid Token Mixer | 减少 full attention 层占比 |
| 2026+ | Compressed / Sparse Attention + MTP + FP8/FP4 | 长上下文、长输出、高吞吐 serving 共设计 |

# 未来 6 到 18 个月趋势预测

## MoE 会成为大模型默认主干

未来大模型大概率继续走 MoE。

原因很简单：

能力提升通常意味着 $P_{\text{total}} \uparrow$。

但推理成本要求：

但推理成本要求 $P_{\text{active}}$ 保持相对可控。

未来可能形成这样的层级：

| 类型 | 可能规模 |
|---|---|
| 旗舰模型 | $1\mathrm{T}+$ total params，$30\mathrm{B}$--$80\mathrm{B}$ active params |
| 高性价比模型 | $200\mathrm{B}$--$500\mathrm{B}$ total params，$5\mathrm{B}$--$20\mathrm{B}$ active params |
| 端侧/小模型 | dense 或小 MoE，$1\mathrm{B}$--$30\mathrm{B}$ params |

判断：

> MoE 会像推荐系统里的大 embedding table / expert pool 一样，成为容量池。

## Full attention 会“贵族化”

未来不会彻底删除 full attention。

但 full attention 可能从“每层都有”变成“少数关键层才有”。

典型 block 可能是：

典型 block 可能是：$[\mathrm{CheapMixer}+\mathrm{MoE}]^3 \rightarrow [\mathrm{GlobalAttention/MLA}+\mathrm{MoE}]$。

full attention 的职责会更聚焦：

- 精确检索。
- 全局校准。
- 长距离 copy。
- 工具调用轨迹对齐。
- 多文档对齐。
- 复杂 ICL。

判断：

> Attention 不会消失，但会变得更少、更贵、更关键。

## Linear 与 Compressed 两条路线会并行

未来 token mixer 不会马上收敛到一个答案。

Linear / recurrent 路线更适合：

- 长上下文。
- 长输出。
- 高 QPS decode。
- KV cache 极度敏感场景。

Compressed / sparse attention 路线更适合：

- 需要较强精确检索能力。
- 需要保留 attention 语义。
- 多文档、多跳推理、复杂 agent 场景。

判断：

> 最终主流可能是二者融合，而不是二选一。

## Memory / State 会成为架构中心

未来看一个 LLM 结构，不能只看层数、hidden size、head 数、expert 数，还要看：

- 每层是否有 KV？
- KV 是否压缩？
- 是否有 recurrent state？
- state 多大？
- 不同层 memory 是否共享？
- prefix cache 如何复用？
- 1M context 下 decode 读多少历史？

判断：

> Memory design 会成为 LLM 架构的第一等公民。

## MTP / Speculative Decoding 会内生到模型结构

未来模型不只是为了 next-token prediction 训练。

越来越多结构会支持：

- 一次预测多个未来 token。
- draft / verifier 配合。
- speculative decoding。
- 更低长输出 wall time。

这对 agent 和 coding 特别重要，因为这类任务输出长、工具调用多、多轮状态长、用户感知延迟强。

判断：

> 未来 LLM 架构会同时优化能力和生成速度。

## Precision / Kernel / Parallel 会变成模型结构的一部分

未来“网络结构”不会只等于 Python 里的 model class。

它还包括：

- FP8 / FP4 训练与推理。
- expert parallel。
- tensor parallel。
- pipeline parallel。
- all-to-all 通信优化。
- cache layout。
- prefix reuse。
- linear attention kernel。
- sparse attention kernel。
- speculative decoding runtime。

判断：

> 模型结构和推理系统会越来越不可分。

# 一个可能的未来主流 LLM block 模板

未来 frontier LLM 可能逐渐接近这种结构：

未来 frontier LLM 可能逐渐接近：Embedding $→ [Cheap\ Mixer+MoE\ FFN]^m → [Global\ Attention/MLA/CSA+MoE\ FFN] → MTP/DraftHead →$ LM Head。

更抽象地写：

更抽象地写：LLM = Sparse Expert Capacity Pool + Hybrid Token Mixer + Compressed Memory + Long-context Objective + Serving Runtime。

# 对 LLM4Rec / 推荐精排的启发

## 纯 full-attention LLM 不是推荐精排长期最优解

推荐精排里的典型问题是：

- 用户历史很长。
- candidate 多。
- QPS 极高。
- latency SLA 很硬。
- 特征和样本流巨大。

如果直接用普通 full-attention dense LLM：

如果直接用普通 full-attention dense LLM，相当于：长 user history $×$ 多 candidate $×$ 高 QPS。

很容易在训练和 serving 上都炸。

因此长期来看，更适合推荐的结构可能不是纯 attention LLM，而是：

因此长期更适合推荐的结构可能是：便宜历史状态压缩 + 少数 target-aware/global attention + 小 active MoE + 强 cache / prefix reuse + distillation 到在线轻量 ranker。

## 多数层做便宜状态更新，少数层做精确交互

推荐系统里可以借鉴 hybrid token mixer 思路：

- 大部分层对用户历史做便宜状态更新 / 压缩。
- 少数层让 target item / candidate 与用户状态做精确交互。

可以抽象为：

可以抽象为：User history tokens $→$ Linear / compressed mixer $→$ User state $→$ Target-aware attention $→$ Ranking head。

这比每一层都做 full user-history $×$ target attention 更适合高 QPS。

## Offline teacher 更适合吸收前沿结构

在线精排受 SLA 限制，不一定能直接部署大 MoE / hybrid long-context LLM。

但 offline teacher 可以：

- 用更长 user history。
- 用更大模型。
- 用更强 attention/global layer。
- 接受分钟级甚至小时级延迟。
- 产出 teacher score / soft label。

然后：

离线蒸馏链路可以写成：Offline frontier LLM teacher $→$ teacher score / soft label $→$ online lightweight ranker。

这可能比直接把大 LLM 塞进在线链路更现实。

## 和 RankMixer 思想的联系

RankMixer 本身受 MLP-Mixer 和 per-token 思想启发。

它和 LLM hybrid token mixer 的共性是：

> 不要一开始把所有信息拍扁，而是保留 token 粒度，再用受控 mixing 做交互。

区别在于：

| 场景 | token 含义 | mixer 目标 |
|---|---|---|
| LLM | 文本 token / SID token / prompt token | 长序列语义建模与生成 |
| RankMixer | 推荐特征 token / field token / item token | 高吞吐排序建模 |

但是上位思想类似：

上位思想是：保留结构化 token $→$ 控制交互成本 $→$ 关键位置做强交互 $→$ 满足工业 serving 约束。

# 最终判断

未来 LLM 网络结构大概率会同时满足以下特征。

## 稀疏化

Channel mixer 的演变可以概括为：Dense FFN $→$ MoE FFN。

总参数继续涨，但 active params 可控。

## 混合化

混合化指的是：Full Attention + Linear/Recurrent Mixer + Compressed/Sparse Attention。

不同层承担不同 token mixing 职责。

## 记忆压缩化

记忆压缩化指的是：Full KV Cache $→$ Latent KV $→$ Compressed KV $→$ Recurrent State。

memory/state 成为架构设计中心。

## 生成加速内生化

生成加速内生化指的是：Next-token head $→$ MTP / draft head / speculative-friendly head。

模型结构开始直接服务生成速度。

## 系统共设计化

系统共设计化指的是：模型结构 + 训练稳定性 + 量化 + 并行 + cache + kernel + serving runtime。

未来模型不再只是一个 checkpoint，而是一整套软硬件共设计产物。

# 最短总结

2023 年的 LLM 还是“更强的 Transformer”；2024 年进入“MoE Transformer”阶段，通过稀疏专家扩展容量；2025 年以后，Qwen3-Next、Kimi Linear、Hunyuan TurboS/T1 等模型开始证明 hybrid token mixer 成为主线，即多数层用 GDN/KDA/Mamba-like 等便宜状态更新，少数层保留 global attention/MLA 兜底；DeepSeek V4 则代表 compressed/sparse attention 路线，说明未来不一定只有 linear 一条路。

长期来看，LLM 会演化成 MoE 容量池、混合 token mixer、压缩 memory/state、MTP/spec decoding、FP8/FP4 和推理系统共同设计的稀疏专家系统。

对于 LLM4Rec，最有价值的启发是：

> 不要指望纯 full-attention LLM 直接在线精排，而应考虑低成本历史状态压缩、少数 target-aware/global attention、offline teacher distillation 和在线轻量 ranker 的组合。
