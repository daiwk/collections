# RankMixer 与序列-特征统一建模论文地图

本文整理两条正在汇合的工业推荐模型路线：

1. 以 Wukong、RankMixer 为代表的非序列特征交互和 dense scaling。
2. 以 HSTU、LONGER、STCA、SlimPer 为代表的长用户行为序列建模。

重点讨论 RankMixer 的直接后续工作，以及 HyFormer、MixFormer、InterFormer、OneTrans、EST、WHALE 等把两条路线放入同一 backbone 的方法。

> 名称提醒：本文中的 [Hiformer](https://arxiv.org/abs/2311.05884) 是 Google 2023 年的异构特征交互模型；[HyFormer](https://arxiv.org/abs/2601.12681) 是 ByteDance 2026 年的长序列与非序列特征统一模型。两者名称相似，但不是同一项工作。

## 1. 核心结论

- RankMixer 本身主要处理固定数量的异构 feature tokens，不是长序列模型。
- 工业生产中的强基线通常是 `长序列模块 -> RankMixer`，例如 `LONGER -> RankMixer` 或 `STCA -> RankMixer`。
- 这种串联方案的问题是序列在进入 RankMixer 前已经被压缩，非序列特征无法反复影响历史读取。
- HyFormer 和 MixFormer 都把 RankMixer 式 token mixing 放到每个长序列读取层中，使高阶 user/item/context 特征直接形成或增强 Query。
- HyFormer 的顺序是 `Query Decoding -> Query Boosting`；MixFormer 的顺序是 `Query Mixer -> Cross Attention -> Output Fusion`。
- SlimPer、HyFormer、MixFormer 和 STCA 都属于 `Q_short x KV_long` 路线，但 Query 的来源和层间状态不同。
- HSTU、MTGR、WHALE 保留更多 history-to-history self-attention，表达能力更强，但序列侧成本更高。
- RankMixer 后续研究已从单纯扩大参数，转向训练稳定性、请求级复用、有效秩、可学习 mixing 和序列-dense 联合扩展。

## 2. 统一问题定义

记：

```text
H: 用户行为序列，长度 L
F: user/item/context/cross 等非序列特征 token，数量 M
Q: 用于读取历史的紧凑 Query，数量 q
D: hidden dimension
C: 同一请求中的候选 item 数
```

工业排序模型需要同时解决三个问题。

### 2.1 长序列建模

需要决定是否执行：

```text
History-to-History:
SelfAttention(H, H, H)
```

或者只执行：

```text
Target/Query-to-History:
CrossAttention(Q, H, H)
```

前者通常具有 `O(L^2 D)` 成本，后者在固定 `q` 时约为 `O(qLD)`。

### 2.2 异构特征交互

非序列特征来自不同语义空间：

```text
user profile
candidate item
request context
cross features
dense statistics
pretrained semantic embeddings
```

直接使用标准 self-attention 时，需要比较来自不同 ID 空间的向量内积。RankMixer 认为这种相似度并不天然可靠，因此用无参数的 token mixing 交换信息，再用 per-token FFN 建模各特征子空间。

### 2.3 序列与非序列特征融合

主要存在四种结构：

```text
Late Fusion:
HistoryEncoder(H) -> history summary
FeatureMixer([F; history summary])

Alternating Fusion:
Q_l reads H
Q_l mixes with F
repeat

Monolithic Fusion:
[H; F] enters one Transformer stream

Dual-Branch Fusion:
Sequence branch and feature-interaction branch remain active
cross-branch fusion happens at every layer
```

## 3. RankMixer 之前的 Dense Scaling 路线

| 论文 | 公司 | 主要交互算子 | 与 RankMixer 的关系 |
|---|---|---|---|
| [DHEN](https://arxiv.org/abs/2203.11014) | Meta | 在层次结构中组合 DCN、self-attention、FM 等多种算子 | 证明堆叠统一 interaction block 可以扩展，但算子较碎片化 |
| [Hiformer](https://arxiv.org/abs/2311.05884) | Google | feature-specific Q/K/V 的 heterogeneous attention | 用参数隔离处理异构 feature spaces，是 attention 路线代表 |
| [Wukong](https://arxiv.org/abs/2403.02545) | Meta | Factorization Machine Block + Linear Compression Block | 证明非序列 feature interaction 存在清晰 dense scaling law |
| [HHFT](https://arxiv.org/abs/2511.20235) | Alibaba | 层次化 heterogeneous feature transformer | attention-based 异构交互的工业扩展 |
| [RankMixer](https://arxiv.org/abs/2507.15551) | ByteDance | parameter-free Token Mixing + Per-token FFN | 以 GPU 友好方式替换 feature-token self-attention |

这条路线的核心目标不是增加行为序列长度，而是让固定数量的 feature tokens 随深度和宽度稳定扩展。

## 4. RankMixer

### 4.1 输入和基本结构

RankMixer 首先将大量 sparse、dense 和交叉特征按语义分组为固定数量的 tokens：

```text
X = [x_1, x_2, ..., x_T] in R^(T x D)
```

每个 RankMixer block 包含：

```text
1. Multi-head Token Mixing
2. Per-token FFN
```

### 4.2 Multi-head Token Mixing

每个 token 按 hidden dimension 分成多个 head，然后将不同 token 的同一 head 拼在一起：

```text
x_t -> [x_t^1 | x_t^2 | ... | x_t^H]

s_h = Concat(x_1^h, x_2^h, ..., x_T^h)
```

当 `H=T` 时，输出 token 数保持不变。这个操作本质上是 reshape、transpose 和 concat：

```text
TokenMix(X) = Reshape(Transpose(Reshape(X)))
```

它有以下特点：

- 不生成 `T x T` attention score matrix。
- 不依赖跨异构 feature spaces 的内积相似度。
- 每个输出 token 都含有所有输入 token 的部分信息。
- mixing 本身无参数，参数容量主要放在后续 FFN。

### 4.3 Per-token FFN

每个 mixed token 使用独立 FFN：

```text
y_t = FFN_t(s_t) + residual
```

目的包括：

- 避免高频 feature space 支配低频和长尾 feature space。
- 让不同 token 拥有独立参数容量。
- 增加参数时不引入 attention matrix 的 I/O 开销。

### 4.4 RankMixer 的优势

- 将 MFU 从论文基线的约 4.5% 提升到约 45%。
- 在大致相同线上延迟下扩大 dense 参数规模。
- TokenMixer 和 PFFN 都适合 GPU 上的大矩阵计算。
- 可以将 PFFN 扩展为 Sparse MoE。
- 很适合作为长序列编码器后的 feature-interaction head。

### 4.5 RankMixer 的限制

- 原始 mixing 是固定 permutation，数据无关且不可学习。
- 直接对 mixing 前后 token 做 residual，可能存在语义位置不对齐。
- 工业版本通常较浅，继续增加深度时训练稳定性变差。
- PFFN 可能持续压缩表示的有效秩，出现 embedding collapse。
- 单独使用时不会保留原始长行为序列，只能接收池化或压缩后的 sequence token。
- 多候选请求中，如果 user/item 特征已混合，用户侧计算难以直接复用。

## 5. RankMixer 的直接后续工作

| 论文 | 公司 | 主要问题 | 核心改动 | 与长序列的关系 |
|---|---|---|---|---|
| [TokenMixer-Large](https://arxiv.org/abs/2602.06563) | ByteDance | RankMixer 深度、residual、MoE 和大规模训练受限 | Mixing-and-Reverting、inter-layer residual、auxiliary loss、small init、Sparse-Pertoken MoE、FP8、Token Parallel | 仍以 dense feature interaction 为主，可作为长序列模块后的大规模 head |
| [Compute Only Once / UG-Sep](https://arxiv.org/abs/2602.10455) | ByteDance | 多候选场景重复计算用户侧 token | 用 mask 和可复用 PertokenAFFN 分离 user-group 信息流，加入信息补偿和 W8A16 | 与 STCA、SlimPer、UI-MixFormer 的 request-level reuse 目标一致 |
| [UniMixer](https://arxiv.org/abs/2604.00590) | Kuaishou | TokenMixer 是固定规则，attention、TokenMixer、FM 缺少统一解释 | 将 TokenMixer 等价为参数矩阵，构造可学习 UniMixing，并提出轻量 UniMixing-Lite | 提供可替换 Query Mixer/feature mixer 的通用算子 |
| [RankElastor](https://arxiv.org/abs/2605.23191) | Tencent / HKUST(GZ) | RankMixer token mixing 扩秩有限，PFFN 导致秩收缩 | Parameterized Full Mixing + GLU-improved P-FFN | 可改善统一序列模型中 compact query 的表示容量 |
| [RankUp](https://arxiv.org/abs/2604.17878) | Tencent | 模型加深后有效秩呈衰减振荡 | 随机 permutation splitting、多 embedding、global token、预训练 token cross、task token 解耦 | 更偏输入表示和多任务 token 设计，可接入序列 encoder 输出 |

### 5.1 TokenMixer-Large

TokenMixer-Large 是 RankMixer/TokenMixer 最直接的工程和模型升级。

它针对的主要问题是：

```text
原始 token 与 mixed token 的 residual 语义不对齐
深层网络梯度更新不足
Dense-Train-Sparse-Infer 不能节省训练成本
1B 左右规模后继续扩展困难
```

主要改动：

- Mixing-and-Reverting 恢复 token 语义后再做 residual。
- Inter-layer residual 和 auxiliary loss 改善深层训练。
- SwiGLU down projection 使用较小初始化。
- Sparse-Pertoken MoE 采用 Sparse-Train-Sparse-Infer。
- Shared expert、gate value scaling 和 token parallel 改善专家训练。
- 移除 DCN、LHUC 等低 MFU 碎片算子，形成 pure backbone。

论文报告离线达到 15B 参数，线上不同场景达到 7B、4B 和 2B 级别。

### 5.2 UG-Sep

UG-Sep 的问题不是进一步增加表达力，而是：

```text
同一个 user/request 有 C 个候选 item
若 user/item token 在 RankMixer 中过早混合
用户侧 FFN 和 interaction 会被重复执行 C 次
```

它通过受控的信息流：

```text
user representation 不能依赖当前 candidate
item representation 可以读取 user representation
```

使用户侧中间结果在请求内复用。这与以下系统思想相同：

- STCA 的 Request Level Batching。
- SlimPer 的 Request-Only Optimization。
- OneTrans 的 cross-candidate / cross-request KV cache。
- UI-MixFormer 的 user-item decoupled HeadMixing。

### 5.3 UniMixer

UniMixer 的关键观察是：固定 TokenMixer 可以表示成一个 permutation matrix：

```text
flatten(TokenMixer(X)) = W_perm * flatten(X)
```

因此可以把固定 `W_perm` 推广成可学习、结构化、低成本的 mixing：

```text
fixed TokenMixer
    -> parameterized mixing
    -> unified view of Attention / TokenMixer / FM
```

UniMixer 更适合被理解为 RankMixer mixing operator 的理论推广，而不是长序列架构。

### 5.4 RankElastor 与 RankUp

这两篇工作都关注：

> 参数数量增加，是否真的带来更高维、更有区分度的表示？

RankElastor 从 block 内部修复：

- 用 Parameterized Full Mixing 增强 mixing 的扩秩能力。
- 用 GLU-improved P-FFN 减弱 FFN 的秩收缩。

RankUp 从输入和 token 组织修复：

- 随机拆分 sparse features，降低 token 间同质性。
- 为同一特征构造多 embedding 表示。
- 引入 global、pretrained 和 task-specific tokens。

两者都说明 RankMixer 的后续研究重点已从“增加参数”转向“确保参数真正增加有效表示容量”。

## 6. RankMixer 的相邻替代路线

| 论文 | 公司 | 交互机制 | 主要观点 |
|---|---|---|---|
| [INFNet](https://arxiv.org/abs/2508.11565) | Kuaishou | 少量 hub tokens 先 aggregate，再 gated broadcast 回全部 token | 保持原始 token 宽度，以线性成本避免 early aggregation |
| [EST](https://arxiv.org/abs/2602.10811) | Alibaba | Lightweight Cross-Attention + Content Sparse Attention | full attention 中最重要的是 non-behavior 与 behavior 的 cross interaction，其余可稀疏化 |
| [Hiformer](https://arxiv.org/abs/2311.05884) | Google | token-specific heterogeneous attention | 异构 feature 应使用各自 Q/K/V 参数 |
| [Wukong](https://arxiv.org/abs/2403.02545) | Meta | stacked FM + linear compression | 高阶显式乘性交互适合 dense scaling |
| [UniMixer](https://arxiv.org/abs/2604.00590) | Kuaishou | generalized parameterized mixing | attention、TokenMixer 和 FM 可以放进统一 mixing 框架 |

INFNet 与 RankMixer 的区别尤其值得注意：

```text
RankMixer:
所有 token 经固定重排交换部分维度

INFNet:
原始 token 保留
hub 聚合全局信息
再把全局信息广播回原始 token
```

因此 INFNet 更接近 Perceiver、SlimPer 的 latent bottleneck，但它额外把信息广播回长 token 流，保留 width-preserving 状态。

## 7. 从 Late Fusion 到 Unified Backbone

### 7.1 工业强基线：序列模块后接 RankMixer

典型结构：

```text
H_long
  -> LONGER / STCA / DIN / HSTU
  -> one or several sequence summaries

[sequence summaries; user/item/context tokens]
  -> RankMixer
  -> prediction heads
```

优点：

- 两个模块可以独立演进。
- 序列模块和 dense 模块可分别调参。
- 系统成熟，容易替换局部模块。

缺点：

- 序列在进入 RankMixer 前已经压缩。
- 非序列特征通常只参与序列 Query 的初始构造。
- RankMixer 学到的高阶特征无法回头改变上一层历史读取。
- 增加序列模块和 dense 模块的参数时，两边竞争同一 FLOPs 预算。
- 共同 scaling 时常出现“dense 扩大收益高但吃不到长序列，sequence 变长收益高但挤压 dense 容量”的矛盾。

## 8. InterFormer：较早的双向交替融合

[InterFormer](https://arxiv.org/abs/2411.09852) 将模型分为：

```text
Interaction Arch: 非序列特征交互
Sequence Arch: 行为序列建模
Cross Arch: 两个分支之间交换信息
```

核心目标：

- 避免 sequence summary 只单向流入 interaction branch。
- 让两个模态在多层中反复互相更新。
- 避免过早把序列压成一个向量。

InterFormer 可以看作 HyFormer、MixFormer、WHALE 之前的重要结构先驱：它已经明确指出 late fusion 和单向信息流的问题，但仍保留较明显的双分支结构。

## 9. HyFormer

论文：[HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)

公司：ByteDance

### 9.1 核心结构

HyFormer 每层交替执行：

```text
1. Query Decoding
2. Query Boosting
```

#### Query Decoding

每个行为序列拥有独立 global query：

```text
Q_hat_l = CrossAttention(Q_(l-1), K_l(H), V_l(H))
```

序列 K/V 可以使用不同后端：

- Full Transformer encoding。
- LONGER-style short-query / long-KV encoding。
- 仅 FFN 的轻量 encoding。

#### Query Boosting

将读取历史后的 global queries 与非序列 tokens 拼接：

```text
U_l = [Q_hat_l; F_1; ...; F_M]
Q_l = RankMixerStyleBoost(U_l)
```

Boost 模块使用：

- MLP-Mixer / RankMixer-style token mixing。
- Per-token FFN。
- 不同序列各自拥有独立 global query。

随后增强后的 Query 进入下一层，再次读取相应长历史。

### 9.2 统一公式

```text
Q_hat_l = CrossAttn(Q_(l-1), K_l(H), V_l(H))
Q_l     = QueryBoost([Q_hat_l; F])
```

其核心信息流是：

```text
历史 -> Query -> 与非序列特征交互 -> 更强 Query -> 再读历史
```

### 9.3 相比 LONGER -> RankMixer

```text
LONGER -> RankMixer:
历史只在前半段读取
RankMixer 学到的交互不能反馈给历史 encoder

HyFormer:
每层读历史后都与非序列特征交互
更新后的 Query 会影响下一层历史读取
```

### 9.4 多序列设计

HyFormer 不主张把观看、搜索、购买等序列简单拼成一个流，而是：

```text
watch sequence    <-> watch global query
search sequence   <-> search global query
purchase sequence <-> purchase global query
```

跨序列信息在 Query Boosting 阶段交互。这能避免不同 side information 和语义空间被强制对齐。

## 10. MixFormer

论文：[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)

公司：ByteDance / Douyin

### 10.1 核心结构

每个 MixFormer block 包含：

```text
1. Query Mixer
2. Cross Attention
3. Output Fusion
```

#### Query Mixer

输入是非序列 feature heads：

```text
X_l = [x_1, ..., x_N]
P_l = HeadMixing(Norm(X_l)) + X_l
Q_l^i = SwiGLU_i(Norm(P_l^i)) + P_l^i
```

这里的 HeadMixing 直接来自 RankMixer 思路：

- 通过 reshape/transpose 交换不同 heads 的子空间。
- 不计算异构 feature tokens 间的 attention score。
- 每个 head 使用独立 FFN。

#### Cross Attention

Query Mixer 输出的每个 head 都作为专门 Query 读取长序列：

```text
z_i = CrossAttention(q_i, K_i(H), V_i(H)) + q_i
```

不同 Query head 对应不同高阶非序列特征子空间。

#### Output Fusion

每个 cross-attention 输出使用独立 per-head SwiGLU：

```text
o_i = SwiGLU_i(Norm(z_i)) + z_i
```

输出进入下一 MixFormer block。

### 10.2 统一公式

```text
Q_l   = QueryMixer(X_l)
Z_l   = CrossAttn(Q_l, K_l(H), V_l(H))
X_l+1 = PerHeadFusion(Z_l)
```

信息流是：

```text
高阶非序列交互 -> 形成多个专门 Query -> 读取历史 -> 融合 -> 重复
```

### 10.3 Co-scaling

MixFormer 的核心实验观察：

- 固定序列长度时，扩大 RankMixer dense component 的边际收益往往高于扩大 sequence component。
- 固定 dense 参数时，STCA 一类序列模型从更长历史中获益更明显。
- 简单的 `STCA + RankMixer` 必须在两类计算预算之间做取舍。
- MixFormer 让同一组 per-head FFN 和层间状态同时服务于 dense interaction 与 sequence aggregation，从而改善 co-scaling。

论文比较的序列长度包括：

```text
512
2,048
8,192
10,000
```

### 10.4 UI-MixFormer

原始 MixFormer 会混合 user/item heads，妨碍请求级复用。UI-MixFormer 将 heads 分为：

```text
user-side heads
item-side heads
```

并使用 mask 保证：

```text
user-side output 不读取 item-side signal
item-side output 可以读取 user-side signal
```

从而使用户侧 Query Mixer 和部分 sequence cross-attention 结果可以在同一请求的候选间共享。

## 11. HyFormer 与 MixFormer 的差别

| 维度 | HyFormer | MixFormer |
|---|---|---|
| Layer 顺序 | Cross-Attention 后做 Query Boosting | Query Mixer 后做 Cross-Attention，再 Output Fusion |
| Query 初始来源 | target、global NS features、sequence pooling tokens | 非序列 features 经 split 和 HeadMixing 得到多个 heads |
| Dense 交互算子 | RankMixer/MLP-Mixer 风格 Query Boost | RankMixer 风格 HeadMixing + per-head SwiGLU |
| 多序列 | 每条序列独立 Query 和 K/V，Query 层面交互 | 论文重点是统一 heads 与同一行为序列的交互 |
| 序列 K/V | 支持 full、LONGER-style、轻量 FFN 等多种编码 | 每层使用独立 sequence FFN 和 K/V projection |
| 层间状态 | global queries + NS tokens | 融合后的 feature-conditioned heads |
| 核心主张 | Query 需要不断吸收异构特征后再读序列 | dense 与 sequence 应共享一套可 co-scale 参数 |
| 请求级优化 | 讨论 KV cache/M-FALCON 等部署约束 | 明确提出 UI-MixFormer 和 request-level sharing |

两者都可以视为以下迭代：

```text
Q_l = DenseFeatureInteraction(Q_l, F)
Q_l = CrossAttention(Q_l, H, H)
```

差别主要是操作顺序、Query 组织方式和参数共享设计。

## 12. 其他 Unified Sequence + Feature 模型

| 论文 | 公司 | 统一方式 | 长序列处理 | Dense 交互 |
|---|---|---|---|---|
| [OneTrans](https://arxiv.org/abs/2510.26104) | ByteDance / NTU | 所有 sequence 和 NS tokens 进入单个 causal Transformer | Query 数随层数逐渐减少，K/V 保留当前完整状态 | mixed parameterization self-attention |
| [TokenFormer](https://arxiv.org/abs/2604.13737) | Tencent | static、behavior、target 进入统一 token stream | 浅层 full causal attention，深层 shrinking sliding window | attention + NLIR 乘性交互 |
| [EST](https://arxiv.org/abs/2602.10811) | Alibaba | 原始 behavior 和 NS tokens 进入统一序列 | LCA 删除低价值交互，CSA 用内容相似度做 top-K sparse attention | token-specific parameterized interaction |
| [INFNet](https://arxiv.org/abs/2508.11565) | Kuaishou | categorical、sequence、task tokens 全部保留 | hub cross-attention 聚合，再 gated broadcast 回原始 tokens | aggregate-and-broadcast |
| [WHALE](https://arxiv.org/abs/2607.17017) | Meta | Wukong branch 与 HSTU branch 每层并行并融合 | HSTU 保留 sequence self-attention | Wukong FM-style interaction |
| [InterFormer](https://arxiv.org/abs/2411.09852) | Meta | interaction branch 和 sequence branch 通过 Cross Arch 交替交换 | 独立 sequence architecture | 独立 interaction architecture |
| [SlimPer](https://arxiv.org/abs/2607.12281) | Meta | 固定 user-item knowledge base 统一查询所有用户侧模态 | 每层固定 Query 读取原始历史，无 history self-attention | 显式 matching + MLP refinement |

### 12.1 EST

EST 的观点是：

```text
完整 unified self-attention 有大量冗余
最重要的是：
1. non-behavior tokens 与 behavior tokens 的 cross interaction
2. behavior 内由内容相似度引导的少量高价值 interaction
```

因此它采用：

- Lightweight Cross-Attention 保留跨模态高价值边。
- Content Sparse Attention 在行为序列内只保留 top-K 相似关系。
- 用户侧行为计算可以跨候选共享。

这条路线位于 MixFormer 和 full self-attention 之间：它保留原始 behavior token 状态，但把 attention graph 结构化稀疏。

### 12.2 INFNet

INFNet 使用：

```text
Hub Aggregate:
small hubs query all feature/sequence/task tokens

Gated Broadcast:
hubs send global context back to original tokens
```

它与 SlimPer 都使用固定 latent bottleneck，但区别是：

```text
SlimPer:
只传播 compact knowledge base

INFNet:
compact hubs 聚合后，再更新完整 token 流
```

所以 INFNet 保留更多 token-level 状态，内存更接近 `O(L)`，但能够在多层中复用细粒度行为信息。

### 12.3 WHALE

WHALE 每层同时保留：

```text
Wukong branch: non-sequence feature interaction
HSTU branch: behavior sequence modeling
Fusion branch: Wukong representation queries HSTU representation
```

它与 HyFormer/MixFormer 的共同点是每层交互；不同点是 WHALE 不试图把两类算子折叠成一套参数，而是让两个成熟 backbone 始终并行存在。

## 13. 与长序列论文的统一对比

| 模型 | 长历史怎么读 | history-to-history SA | 非序列特征交互 | 融合时机 | 层间状态规模 | 典型复杂度 |
|---|---|---|---|---|---|---|
| RankMixer | 不直接读取原始长历史 | 无 | TokenMix + PFFN | 通常接收序列 summary，late fusion | `O(MD)` | 与固定 feature token 数相关 |
| LONGER -> RankMixer | LONGER 首层短 Q 读长 KV，后续短序列 SA | 在压缩 Q 长度上有 | RankMixer | 序列编码后 | `O(kD + MD)` | `O(kL + k^2)` 加 dense mixer |
| STCA -> RankMixer | target Q 每层读取长历史 | 无 | RankMixer | sequence 完成后 late fusion | sequence 侧固定，dense 侧固定 | `O(d_s qL)` |
| HyFormer | global Q 每层读取相应序列 | 可配置，主高效模式可无 | Query Boosting | 每层交替 | `O((q+M)D)` | `O(d_s qL)` 加 mixer |
| MixFormer | 高阶 feature heads 每层读取历史 | 无 | Query Mixer + per-head FFN | 每层统一 | `O(MD)` | `O(d_s ML)` 加 mixer |
| SlimPer | knowledge slots 每层读取全部用户侧 token | 无 | matching + refinement | 每层统一 | 固定 `O(KD)` | `O(d_s qL)` |
| OneTrans | 统一 causal stream，Query 逐层减少 | 浅层有，之后逐层压缩 | self-attention | 单流每层 | 随层缩短 | 介于 full SA 与 cross-attention |
| EST | 原始 tokens + 结构化稀疏 graph | top-K 内容稀疏 SA | LCA | 单流每层 | `O(LD)` | 对 L 近似线性 |
| INFNet | hubs 读全部 tokens，再广播回去 | 无 dense SA | hub aggregate/broadcast | 每层统一 | `O(LD)` | 对 token 数线性 |
| HSTU / MTGR | 完整窗口 causal self-attention | 有 | self-attention / DLRM features | 单流或深度融合 | `O(LD)` 每层激活 | `O(L^2D)` |
| WHALE | HSTU 序列分支 | 有 | Wukong | 每层双分支融合 | 两个分支均保留 | HSTU SA + Wukong |

其中：

```text
L: 历史长度
M: 非序列 feature/query token 数
q: compact query 数
K: SlimPer knowledge slots
d_s: 层数
```

## 14. 从 Query 角度对比

### 14.1 STCA

```text
Q_l = target state
Q_(l+1) = CrossAttn(Q_l, H, H)
```

Query 主要围绕候选 target。

### 14.2 SlimPer

```text
Q_l = Project(KnowledgeBase_l)
R_l = CrossAttn(Q_l, all user-side tokens)
KnowledgeBase_(l+1) = MatchAndRefine(KnowledgeBase_l, R_l)
```

Query 是固定多槽位 user-item relevance state。

### 14.3 HyFormer

```text
Q_hat_l = CrossAttn(Q_(l-1), H_l, H_l)
Q_l = RankMixerStyleBoost([Q_hat_l; F])
```

Query 先读取历史，再与所有异构非序列特征交互。

### 14.4 MixFormer

```text
Q_l = RankMixerStyleQueryMixer(F_l)
Z_l = CrossAttn(Q_l, H_l, H_l)
F_(l+1) = PerHeadFusion(Z_l)
```

Query 先由高阶 feature interaction 生成，再读取历史。

### 14.5 WHALE

```text
F_l = Wukong(F_(l-1))
H_l = HSTU(H_(l-1))
F_(l+1) = Fusion(query=F_l, key/value=H_l)
```

Query 来自独立 dense branch，历史来自保留 self-attention 的 sequence branch。

## 15. Full Self-Attention 边界

| 模型 | 是否在长历史上做 full self-attention | 说明 |
|---|---|---|
| RankMixer | 否 | 只处理固定 feature tokens |
| HyFormer | 可选 | 支持 full sequence encoder，但主打高效 cross-attention 配置 |
| MixFormer | 否 | 每层只执行 feature-head-to-history cross-attention |
| STCA | 否 | target-to-history cross-attention |
| SlimPer | 否 | knowledge-slot-to-history cross-attention |
| LONGER | 部分 | 第一层长 K/V cross-attention，后续只在短 Q 上 self-attention |
| OneTrans | 部分 | 初始层接近 full causal attention，之后 Query 逐层缩短 |
| TokenFormer | 部分 | 浅层 full，深层 sliding window |
| EST | 否 | LCA + top-K Content Sparse Attention |
| INFNet | 否 | hubs cross-attention + broadcast |
| HSTU | 是 | 完整输入窗口 causal self-attention |
| MTGR | 基本是 | HSTU 风格统一序列建模 |
| WHALE | 是 | HSTU branch 保留 full sequence modeling |

## 16. 三条主要技术路线

### 16.1 Dense-first

代表：

- RankMixer。
- TokenMixer-Large。
- UniMixer。
- RankElastor。
- RankUp。

核心：

```text
先把固定 feature token interaction 做大、做深、做得 GPU 友好
序列通过 summary token 接入
```

适合：

- 丰富 sparse/cross features 非常强。
- 序列长度固定或序列已有成熟 encoder。
- 线上瓶颈主要是 dense interaction 和 MFU。

### 16.2 Sequence-first

代表：

- HSTU / MTGR。
- LONGER。
- STCA。
- SlimPer。

核心：

```text
先解决长历史读取或序列推理
dense features 作为 Query、context 或最终 head 输入
```

适合：

- 长历史是主要增益来源。
- 业务依赖实时兴趣和行为顺序。
- 可以接受较弱 dense interaction，或有独立 dense head。

### 16.3 Co-scaling / Unified

代表：

- InterFormer。
- HyFormer。
- MixFormer。
- OneTrans。
- EST。
- INFNet。
- WHALE。

核心：

```text
长序列建模和 dense feature interaction 不再只在末端拼接
两类信息在每层或多个阶段反复交换
```

它们内部又分为：

```text
Compact-query unified:
HyFormer, MixFormer, SlimPer

Monolithic token-stream:
OneTrans, TokenFormer, EST, MTGR

Dual-branch progressive fusion:
InterFormer, WHALE

Hub aggregate-and-broadcast:
INFNet
```

## 17. 设计取舍

### 17.1 为什么不用 self-attention 处理所有 feature tokens

RankMixer、HyFormer 和 MixFormer 的共同观点：

- NLP token 通常共享词向量空间，内积相似度有明确含义。
- 推荐中的 user ID、item ID、context、统计特征来自不同空间。
- 对这些 token 使用共享 Q/K/V 可能造成语义错配。
- attention score matrix 增加 I/O 和 memory-bound 开销。
- parameter-free mixing + token/head-specific FFN 更适合固定数量异构特征。

### 17.2 为什么 cross-attention 仍然适合读行为序列

MixFormer 明确区分：

```text
feature-to-feature:
异构空间，内积相似度不稳定

structured-query-to-history:
目标是相关性检索，attention 内积有明确作用
```

因此它在 Query Mixer 中不用 self-attention，但在 Query 读取历史时保留 cross-attention。

### 17.3 为什么统一模型仍然要做 user-item decoupling

完全混合 user/item features 会带来：

```text
更强的表达能力
但用户侧状态依赖候选 item
导致同请求 C 个候选重复执行用户侧计算
```

所以生产统一模型通常加入单向信息流：

```text
user state 不读取 item
item state 可以读取 user
```

代表：

- UG-Sep。
- UI-MixFormer。
- OneTrans causal ordering。
- STCA RLB。
- SlimPer ROO。
- EST user-candidate decoupled computation。

## 18. 推荐阅读顺序

### 18.1 RankMixer 主线

1. [Wukong](https://arxiv.org/abs/2403.02545)
2. [RankMixer](https://arxiv.org/abs/2507.15551)
3. [TokenMixer-Large](https://arxiv.org/abs/2602.06563)
4. [Compute Only Once / UG-Sep](https://arxiv.org/abs/2602.10455)
5. [UniMixer](https://arxiv.org/abs/2604.00590)
6. [RankElastor](https://arxiv.org/abs/2605.23191)
7. [RankUp](https://arxiv.org/abs/2604.17878)

### 18.2 从 RankMixer 走向长序列统一

1. [LONGER](https://arxiv.org/abs/2505.04421)
2. [RankMixer](https://arxiv.org/abs/2507.15551)
3. [InterFormer](https://arxiv.org/abs/2411.09852)
4. [HyFormer](https://arxiv.org/abs/2601.12681)
5. [MixFormer](https://arxiv.org/abs/2602.14110)
6. [EST](https://arxiv.org/abs/2602.10811)
7. [WHALE](https://arxiv.org/abs/2607.17017)

### 18.3 对比 compact query 与 full sequence

1. [STCA](https://arxiv.org/abs/2511.06077)
2. [SlimPer](https://arxiv.org/abs/2607.12281)
3. [HyFormer](https://arxiv.org/abs/2601.12681)
4. [MixFormer](https://arxiv.org/abs/2602.14110)
5. [HSTU](https://arxiv.org/abs/2402.17152)
6. [ULTRA-HSTU](https://arxiv.org/abs/2602.16986)

## 19. 简要判断

- **RankMixer 最直接的后续**：TokenMixer-Large。
- **RankMixer 的请求级推理优化**：UG-Sep。
- **RankMixer mixing 的理论推广**：UniMixer。
- **RankMixer 有效秩修复**：RankElastor、RankUp。
- **RankMixer + LONGER 的逐层交替版本**：HyFormer。
- **RankMixer Query head 直接读取长历史的统一版本**：MixFormer。
- **最像 SlimPer 的 dense-enhanced 版本**：HyFormer 和 MixFormer。
- **保留完整原始 token 但避免 full attention**：EST、INFNet。
- **同时保留成熟 dense 和 sequence backbone**：WHALE。
- **完整 history-to-history 表达力上限路线**：HSTU、MTGR、WHALE。
- **最强工业共识**：序列和 dense feature 不能只做末端拼接，但完全统一后必须重新解决多候选计算复用问题。

