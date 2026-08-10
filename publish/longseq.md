# 工业推荐长序列建模论文地图

本文整理工业推荐、广告和搜索场景中的长用户行为序列建模工作，重点关注以下问题：

- 少量 Query 如何读取长历史序列。
- 是否保留 history-to-history self-attention。
- 长历史是在在线阶段直接建模，还是先检索、压缩或缓存。
- 哪些工作真正对完整建模窗口执行 dense full self-attention。

## 1. 统一视角

大量工业长序列模型可以抽象为：

```text
Q_short 读取 K/V_long
```

不同方法的主要区别是：

1. `Q_short` 如何构造：候选 item、采样历史 token、固定 latent slots，或者压缩后的兴趣表示。
2. `K/V_long` 是否保留原始历史，还是经过检索、分组、聚类或 token merge。
3. 每一层是否重新读取长历史。
4. 是否保留 history-to-history self-attention。
5. 用户侧计算能否在多个候选 item 之间共享。

与 SlimPer 最接近的是 Douyin 的 STCA，而 LONGER 位于 cross-attention 和完整序列 self-attention 之间。

## 2. 技术路线总览

| 路线 | 代表论文 / 公司 | 核心结构 | 与 SlimPer 的关系 |
|---|---|---|---|
| Target Attention 起点 | [DIN](https://arxiv.org/abs/1706.06978)，Alibaba | 候选 item 作为 Query，历史作为 K/V | 单层、单个或少量 Query 的原型 |
| 检索后 Target Attention | [SIM](https://arxiv.org/abs/2006.05639)、[ETA](https://arxiv.org/abs/2209.12212)，Alibaba；[TWIN](https://arxiv.org/abs/2302.02352)、[TWIN-V2](https://arxiv.org/abs/2407.16357)，Kuaishou | 先用候选 item 从 `10^4` 到 `10^6` 历史中检索 Top-K，再执行精确 attention | Query 同样依赖 item，但未召回的历史不会进入后续模型 |
| 直接堆叠 Cross-Attention | [STCA](https://arxiv.org/abs/2511.06077)，ByteDance / Douyin | 每层都让 target Query 读取 10k 历史 K/V，不做 history self-attention | 与 SlimPer 最接近 |
| 固定 Latent / Knowledge Slots | [SlimPer](https://arxiv.org/abs/2607.12281)，Meta | 固定 `K=64` 个 knowledge slots，产生约 `q=16` 个 Query，每层重新读取原始历史 | 可看作 STCA 的多槽位、全模态、显式 matching/refinement 版本 |
| 首层 Cross，后续短序列 Self-Attention | [LONGER](https://arxiv.org/abs/2505.04421)，ByteDance | 第一层执行 `(m+k) Query x (m+L) KV`，之后在长度 `m+k` 上执行 self-attention | 保留 sampled history token 间交互，但中间长度仍随历史增长 |
| 逐层缩短 Query | [OneTrans](https://arxiv.org/abs/2510.26104)，ByteDance / NTU | 每层 K/V 覆盖当前全序列，只让越来越少的尾部 token 发出 Query | 位于 LONGER 和 SlimPer 之间 |
| 分段压缩再 Target Attention | [LASER](https://arxiv.org/abs/2602.11562)，Xiaohongshu | 每段用 target-aware attention 压缩，再对 segment 表示执行堆叠 attention | 先局部压缩，避免每层读取全部原始 token |
| Item-independent 摘要缓存 | [VISTA](https://arxiv.org/abs/2510.22049)，Meta | 百万级历史先压成几百个 summary tokens，在线候选 item 再 attention summary | 历史摘要可跨候选缓存，但摘要本身不依赖候选 |
| Offline 用户 Memory | [DV365](https://arxiv.org/abs/2506.00450)，Instagram；[MARM](https://arxiv.org/abs/2411.09425)，Kuaishou | 将数万条历史离线压成 embedding 或 memory | 在线成本低，但 candidate-aware 能力相对较弱 |
| 层次兴趣 Agent | [HiSAC](https://arxiv.org/abs/2602.21009)，Alibaba | 稀疏激活少量 interest agents，再由 agents 聚合历史 | 与固定 latent slots 相似，但强调语义层次和稀疏路由 |

## 3. 结构演化

```text
DIN
  |
  +-- SIM / ETA / TWIN
  |     先找相关历史，再执行精确 target attention
  |
  +-- LONGER
  |     短 Query 读取长历史，然后在短序列上继续 self-attention
  |
  +-- STCA
  |     每一层都只让 target Query 重新读取长历史
  |
  +-- SlimPer
        多个固定 user-item knowledge slots 反复读取原始多模态历史
```

另一条与上述路线平行的发展路径是：

```text
BST / SASRec
  |
  +-- HSTU / MTGR
  |     保留完整窗口上的 history-to-history self-attention
  |
  +-- LONGER / OneTrans
  |     先压缩 Query 长度，再在较短表示上继续建模
  |
  +-- STCA / SlimPer
        放弃 history-to-history self-attention，只保留 relevance-focused cross-attention
```

## 4. LONGER、STCA 与 SlimPer

### 4.1 LONGER

LONGER 的第一层可以写成：

```text
完整输入 R = [global tokens; full or merged history]
长度 = m + L

第一层 Query O = [global tokens; sampled history tokens]
长度 = m + k

Cross-Attention:
Q:   (m+k) x d
K,V: (m+L) x d
输出: (m+k) x d
```

之后的 self-attention 在第一层输出上执行：

```text
Q = K = V = 第一层 Cross-Attention 输出
长度 = m + k
Attention Matrix = (m+k) x (m+k)
```

对应复杂度：

```text
第一层 Cross-Attention: O((m+k)(m+L)d)
后续每层 Self-Attention: O((m+k)^2 d)
```

LONGER 的 `k` 通常包含采样的历史 token，因此可能随历史长度增长，并不是固定数量的 latent slots。

### 4.2 STCA

STCA 用候选 target 表示作为短 Query，并在每层重新读取长历史：

```text
Q_l = target state at layer l
K,V = full user history
Q_(l+1) = CrossAttention(Q_l, K, V)
```

它不维护长度为 `N` 的中间历史状态，也不执行 history-to-history self-attention，因此每层复杂度对历史长度近似线性。

### 4.3 SlimPer

SlimPer 将单个 target state 扩展为固定大小的 user-item knowledge base：

```text
X_l: K 个固定 knowledge slots
Q_l = Project(X_l), Q_l 的长度为 q
K,V = 全部原始用户侧 token
R_l = CrossAttention(Q_l, K, V)
X_(l+1) = MatchAndRefine(X_l, R_l)
```

生产配置中常见：

```text
Knowledge base size K = 64
Query size q = 16
History length N = 10^3 到 10^4+
```

SlimPer 与 STCA 的共同点：

- 每层重新读取长历史。
- 不传播长度为 `N` 的历史 hidden states。
- 主要计算集中于 user-item relevance。

SlimPer 相比 STCA 的新增设计：

- 多个固定 knowledge slots，而不是单一 target state。
- 显式 relevance matching 和 MLP refinement。
- sparse、dense、sequence 等全模态统一建模。
- 用户侧 token 在同请求多个候选之间共享。

## 5. Full Self-Attention 工作

需要区分两种“完整序列”：

1. 对模型实际输入窗口执行完整 `N x N` attention。
2. 对用户生命周期原始历史执行完整 attention，例如 `10k` 到 `1M` 条行为。

第一种在工业系统中存在；第二种由于计算和 I/O 成本极高，非常少见。

| 论文 | Full Self-Attention 情况 | 实际边界 |
|---|---|---|
| [BST](https://arxiv.org/abs/1905.06874)，Alibaba | 是 | 对完整输入窗口做 self-attention，但生产配置长度约为 20，不属于超长序列 |
| [TransAct](https://arxiv.org/abs/2306.00248)，Pinterest | 是 | 实时序列约为 100 |
| [TransAct V2](https://arxiv.org/abs/2506.02267)，Pinterest | 输入 Transformer 后是 full self-attention | 最终 Transformer 输入约为 192，长历史在此前经过 NN 检索 |
| [HSTU](https://arxiv.org/abs/2402.17152)，Meta | 是，典型代表 | 在完整输入窗口上执行二次复杂度 causal self-attention，论文展示到 8192；后续系统工作扩展到约 16k |
| [MTGR](https://arxiv.org/abs/2505.18654)，Meituan | 基本是 | 基于 HSTU，使用 full、autoregressive 和隔离等动态 mask，但输入经过用户级组织和压缩 |
| [GRAB](https://arxiv.org/abs/2602.01865)，Baidu | 不是纯 full | 使用 causal self-attention，但生产结构加入 dual sliding-window 和多通道拆分 |
| [OneTrans](https://arxiv.org/abs/2510.26104) | 部分 | 浅层接近 full causal attention，之后 Query 数量逐层减少 |
| [TokenFormer](https://arxiv.org/abs/2604.13737)，Tencent | 部分 | 浅层 full causal attention，深层使用 shrinking sliding window |
| [ULTRA-HSTU](https://arxiv.org/abs/2602.16986)，Meta | 否 | 从原始 HSTU 转向 semi-local attention |

### 5.1 为什么工业界很少对 10k+ 原始历史做全层 Full Attention

主要瓶颈包括：

- 每层 attention 计算量为 `O(N^2 d)`。
- 训练阶段每层需要维护长度为 `N` 的中间激活。
- 一个请求通常有多个候选 item，若无法复用用户侧状态，会重复计算和复制历史。
- 超长历史的在线读取本身存在显著 I/O 和网络开销。
- 历史越长，噪声和弱相关行为越多，并不保证所有 token-token 交互都有价值。

因此工业方案通常选择以下一种或多种策略：

- 检索 Top-K 历史。
- 将历史分组、聚类或 token merge。
- 用固定 latent slots 读取历史。
- 只在浅层做 full attention。
- 使用 sliding-window、semi-local 或 sparse attention。
- 离线生成用户 memory，并在在线阶段复用。
- 对同一请求的多个候选共享用户侧编码或 KV cache。

## 6. 关键代表论文

### 6.1 Target-aware 检索路线

#### DIN

- 论文：[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- 公司：Alibaba
- 贡献：候选 item 作为 Query，对用户历史执行自适应 target attention。
- 意义：后续 SIM、ETA、TWIN、STCA 和 SlimPer 都可以追溯到这一基本计算模式。

#### SIM

- 论文：[Search-based User Interest Modeling with Lifelong Sequential Behavior Data](https://arxiv.org/abs/2006.05639)
- 公司：Alibaba
- 历史规模：最高约 54,000。
- 贡献：GSU 从超长历史检索候选相关子序列，ESU 再执行精确兴趣建模。
- 局限：检索阶段和精排阶段可能存在目标不一致。

#### ETA

- 论文：[Efficient Long Sequential User Data Modeling for CTR Prediction](https://arxiv.org/abs/2209.12212)
- 场景：Taobao
- 贡献：使用 hashing 和低成本 bit-wise 操作近似 target attention，实现端到端长历史检索。

#### TWIN / TWIN-V2

- 论文：[TWIN](https://arxiv.org/abs/2302.02352)、[TWIN-V2](https://arxiv.org/abs/2407.16357)
- 公司：Kuaishou
- 历史规模：TWIN 面向 `10^4` 到 `10^5`；TWIN-V2 扩展到约 `10^6`。
- 贡献：让 GSU 和 ESU 使用一致的 target-behavior 相关性度量；V2 进一步使用层次聚类压缩生命周期行为。

### 6.2 End-to-end Cross-Attention 路线

#### LONGER

- 论文：[LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/abs/2505.04421)
- 公司：ByteDance
- 贡献：global tokens、token merge、首层 cross-attention、后续短序列 self-attention，以及 KV cache 等系统优化。
- 定位：保留一部分 token-token 交互，同时降低完整 self-attention 成本。

#### STCA

- 论文：[Make It Long, Keep It Fast](https://arxiv.org/abs/2511.06077)
- 公司：ByteDance / Douyin
- 历史规模：10k。
- 贡献：Stacked Target-to-History Cross Attention，用多层 target-to-history cross-attention 替代 history self-attention。
- 系统设计：Request Level Batching 在同一用户请求的多个 target 间共享用户侧编码。

#### SlimPer

- 论文：[SlimPer: Make Personalization Model Slim and Smart](https://arxiv.org/abs/2607.12281)
- 公司：Meta / Instagram
- 历史规模：10k+。
- 贡献：固定大小 user-item knowledge base、逐层查询原始多模态用户证据、显式 matching/refinement，以及全模态 request-only optimization。

#### LASER

- 论文：[LASER: An Efficient Target-Aware Segmented Attention Framework](https://arxiv.org/abs/2602.11562)
- 公司：Xiaohongshu
- 贡献：Segmented Target Attention 先在段内进行 target-aware 压缩，再用 Global Stacked Target Attention 建模跨段关系。
- 定位：介于直接读取完整历史和预压缩之间。

### 6.3 Memory / Summary 路线

#### MIMN

- 论文：[Practice on Long Sequential User Behavior Modeling for CTR Prediction](https://arxiv.org/abs/1905.09248)
- 公司：Alibaba
- 贡献：使用 Multi-channel user Interest Memory Network 增量维护固定大小兴趣 memory，并与 UIC 服务系统协同设计。
- 历史意义：较早的工业长序列固定 memory 方案。

#### MARM

- 论文：[MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation](https://arxiv.org/abs/2411.09425)
- 公司：Kuaishou
- 贡献：缓存复杂模块的中间结果，以较低在线 FLOPs 扩展多层序列建模。

#### DV365

- 论文：[DV365: Extremely Long User History Modeling at Instagram](https://arxiv.org/abs/2506.00450)
- 公司：Meta / Instagram
- 历史规模：平均约 40,000，最高约 70,000。
- 贡献：使用 Multi-slicing and Summarize 策略离线生成稳定的长期用户 embedding，并复用于多个下游模型。

#### VISTA

- 论文：[Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders](https://arxiv.org/abs/2510.22049)
- 公司：Meta
- 历史规模：最高约 1,000,000。
- 贡献：先将历史总结成几百个 token 并缓存，再让候选 item attention 这些 summary tokens。
- 定位：将昂贵的历史建模与候选相关在线计算分离。

### 6.4 Full / Near-Full Self-Attention 路线

#### HSTU

- 论文：[Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152)
- 公司：Meta
- 贡献：将推荐重写为 sequential transduction，使用针对推荐数据优化的完整 causal self-attention。
- 特点：明确承担 `O(N^2)` attention 成本，以保留 history-to-history 高阶交互。

#### MTGR

- 论文：[MTGR: Industrial-Scale Generative Recommendation Framework in Meituan](https://arxiv.org/abs/2505.18654)
- 公司：Meituan
- 贡献：基于 HSTU，同时保留 DLRM 的稀疏和交叉特征，引入 Group Layer Normalization 和动态 mask。

#### GRAB

- 论文：[GRAB: An LLM-Inspired Sequence-First CTR Prediction Modeling Paradigm](https://arxiv.org/abs/2602.01865)
- 公司：Baidu
- 贡献：序列优先的广告 CTR 架构，使用 action-aware causal attention、相对偏置和多通道建模。
- 注意：生产版本使用 sliding-window，因此不是严格的全层 dense full attention。

#### ULTRA-HSTU

- 论文：[Bending the Scaling Law Curve in Large-Scale Recommendation Systems](https://arxiv.org/abs/2602.16986)
- 公司：Meta
- 贡献：系统比较 self-attention 与 cross-attention，认为 self-attention 仍有更强表达能力；同时通过 semi-local attention 等设计改善扩展效率。
- 价值：适合用来理解 STCA/SlimPer 节省计算后可能损失了什么。

## 7. 核心技术争论

这批工作的关键分歧可以概括为：

> history-to-history 交互带来的表达能力，是否值得从 `O(KN)` 回到接近 `O(N^2)` 的成本？

### 支持 Full Self-Attention 的观点

- 行为之间存在顺序、共现、重复、周期和高阶依赖。
- candidate-to-history attention 只能回答“哪些行为和当前 item 相关”，不一定能先推理行为之间的关系。
- HSTU 和 ULTRA-HSTU 的实验认为，在足够算力和系统优化下，self-attention 仍有更好的质量上限。

### 支持 Short Query x Long K/V 的观点

- 排序最终只需要少量 user-item relevance 表示，不需要输出每个历史 token 的预测。
- 大量 history-to-history 交互与当前候选无关，计算利用率低。
- 固定 Query 数量使深度、激活内存与历史长度解耦。
- 同请求多个候选可以共享用户侧 token 或 K/V。
- 在 10k 以上历史规模，线性复杂度通常更容易满足工业延迟和成本约束。

### 折中方案

- LONGER：首层读取长历史，后续在较短 sampled token 上 self-attention。
- OneTrans：K/V 保持完整，Query 数量逐层减少。
- LASER：段内 target-aware 压缩，段间继续建模。
- TokenFormer：浅层 full attention，深层 sliding-window attention。
- VISTA：离线执行候选无关摘要，在线执行候选相关 attention。

## 8. 推荐阅读顺序

### 8.1 理解 `Q_short x KV_long`

1. [DIN](https://arxiv.org/abs/1706.06978)
2. [TWIN](https://arxiv.org/abs/2302.02352)
3. [LONGER](https://arxiv.org/abs/2505.04421)
4. [STCA](https://arxiv.org/abs/2511.06077)
5. [SlimPer](https://arxiv.org/abs/2607.12281)
6. [VISTA](https://arxiv.org/abs/2510.22049)

### 8.2 理解为什么保留 Full Self-Attention

1. [BST](https://arxiv.org/abs/1905.06874)
2. [HSTU](https://arxiv.org/abs/2402.17152)
3. [MTGR](https://arxiv.org/abs/2505.18654)
4. [ULTRA-HSTU](https://arxiv.org/abs/2602.16986)

### 8.3 理解工业系统取舍

1. [SIM](https://arxiv.org/abs/2006.05639)
2. [TWIN-V2](https://arxiv.org/abs/2407.16357)
3. [DV365](https://arxiv.org/abs/2506.00450)
4. [STCA](https://arxiv.org/abs/2511.06077)
5. [VISTA](https://arxiv.org/abs/2510.22049)
6. [LASER](https://arxiv.org/abs/2602.11562)

## 9. 简要判断

- **最像 SlimPer**：STCA。
- **SlimPer 与 LONGER 的中间形态**：OneTrans。
- **坚持 history-to-history 交互**：HSTU、MTGR。
- **研究 cross-attention 表达力损失**：ULTRA-HSTU。
- **面向百万级生命周期历史**：TWIN-V2、VISTA。
- **强调最低在线成本**：DV365、VISTA、MARM。
- **强调候选相关性而不是完整序列推理**：DIN、SIM、TWIN、STCA、SlimPer。

