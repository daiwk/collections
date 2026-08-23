# LLM Prefill 与 Decode FLOPs 近似估算

本文给出一套可以统一估算标准 Transformer、MoE、Linear Attention、Sliding-window Attention 和一般 Sparse Attention 的近似 FLOPs 公式。

核心结论是：

- 稠密 Transformer 常用的 $2N_{\text{params}}L$，只覆盖“参数矩阵乘法随 token 数线性增长”的主体计算。
- Prefill 还要补上 attention 的序列交互项；对 dense attention，它通常是 $O(L^2d)$。
- 带 KV cache 的 decode 不会重算全部历史 token。应拆成“当前 token 经过参数矩阵”和“当前 query 读取历史 KV”两部分。
- MoE 使用每个 token 的激活参数量，而不是模型总参数量。
- Window、sparse 和 linear attention 的区别，主要体现在每个 query 实际交互的历史范围或状态维度。

以下按一次乘加计作 2 FLOPs，即一次乘法和一次加法各计 1 FLOP。公式只估算 forward。

## 1. 统一记号

- $B$：batch size
- $L$：prefill 输入序列长度
- $T$：decode 时已经缓存的历史长度
- $G$：本轮 decode 的 query token 数；单 token decode 时 $G=1$
- $d$：hidden size
- $d_{ff}$：FFN 中间维度
- $n$：Transformer 层数
- $h_q$：query head 数
- $h_{kv}$：KV head 数
- $d_h$：head dimension
- $d_q=h_qd_h$，通常 $d_q=d$
- $d_{kv}=h_{kv}d_h$
- $V$：词表大小
- $W$：local/sliding-window attention 的窗口长度
- $\rho$：稀疏 attention 保留下来的 key 比例
- $k$：MoE top-k experts
- $E$：expert 总数
- $r$：linear attention 的 feature/state dimension

## 2. 标准 Dense Transformer

一层 Transformer 的主要计算包括：

1. Q、K、V 和输出 projection
2. attention score：$QK^\top$
3. attention value aggregation：$\operatorname{softmax}(QK^\top)V$
4. FFN/MLP

下面忽略 normalization、激活函数、softmax、RoPE 等相对较小的项。

### 2.1 单层 Prefill FLOPs

QKV 和输出投影：

$$
F_{\text{proj}}
\approx
2BL\left(dd_q+2dd_{kv}+d_qd\right)
$$

若采用标准 MHA，$d_q=d_{kv}=d$：

$$
F_{\text{proj}}\approx8BLd^2
$$

attention 的两个主要矩阵乘：

$$
F_{\text{attn}}\approx4BL^2d_q
$$

普通两层 FFN：

$$
F_{\text{ffn}}\approx4BLdd_{ff}
$$

SwiGLU/GEGLU 包含三个线性矩阵：

$$
F_{\text{ffn}}\approx6BLdd_{ff}
$$

因此，标准 MHA + SwiGLU 的单层 prefill 近似为：

$$
\boxed{
F_{\text{layer,prefill}}
\approx
8BLd^2+6BLdd_{ff}+4BL^2d
}
$$

全模型为：

$$
\boxed{
F_{\text{prefill}}
\approx
nB\left(8Ld^2+6Ldd_{ff}+4L^2d\right)
+2BLdV
}
$$

最后的 $2BLdV$ 是对所有位置执行全词表 LM head 的计算。如果只对最后一个位置计算 logits，则改为 $2BdV$。

### 2.2 与 $2N_{\text{params}}L$ 的关系

模型主要线性层的参数量约为：

$$
N_{\text{linear}}
\approx
n\left(4d^2+3dd_{ff}\right)
$$

所以：

$$
2BLN_{\text{linear}}
\approx
nB\left(8Ld^2+6Ldd_{ff}\right)
$$

于是一个实用的修正版是：

$$
\boxed{
F_{\text{prefill}}
\approx
2BLN_{\text{active}}
+4nBL^2d
}
$$

也就是：

> 参数矩阵 FLOPs = 2 × 激活参数量 × token 数；然后再加上 attention 的序列交互 FLOPs。

$2N_{\text{params}}L$ 只包含第一部分。

### 2.3 带 KV Cache 的 Decode FLOPs

假设已经缓存 $T$ 个历史 token，本次处理 $G$ 个新 token。

新 token 经过线性层：

$$
F_{\text{linear}}\approx2BGN_{\text{active}}
$$

新 query 与历史 KV 交互：

$$
F_{\text{history-attn}}\approx4BGTd_q
$$

本轮 $G$ 个新 token 之间还会发生 causal self-attention：

$$
F_{\text{within-chunk}}\approx2BG^2d_q
$$

所以：

$$
\boxed{
F_{\text{decode-step}}
\approx
2BGN_{\text{active}}
+4nBGTd_q
+2nBG^2d_q
+F_{\text{lm-head}}
}
$$

单 token decode，即 $G=1$：

$$
\boxed{
F_{\text{decode/token}}
\approx
2BN_{\text{active}}
+4nBTd
+2BdV
}
$$

可以把它理解为两张账单：

- $2N_{\text{active}}$：当前新 token 经过所有参数矩阵一次。
- $4nTd$：当前 query 在每一层读取并匹配 $T$ 条历史 KV。

因此，带 KV cache 的 decode 不能写成 $2N_{\text{params}}T$。历史长度只增加 attention 成本，不会让所有 FFN 和 projection 对历史 token 重新计算。

如果 embedding 与 LM head 权重共享，参数统计里这部分权重可能只出现一次，但 vocab projection 依然会实际执行，不能因为权重共享而删掉对应 FLOPs。

## 3. MHA、GQA 与 MQA

GQA/MQA 减少的是 K/V projection 的计算量以及 KV cache 的尺寸，但 attention 核心点积通常仍由 query head 数决定。

一层 projection：

$$
\boxed{
F_{\text{proj}}
\approx
2BL\left(2d^2+2dd_{kv}\right)
=4BLd^2+4BLdd_{kv}
}
$$

其中：

$$
d_{kv}=h_{kv}d_h
$$

- MHA：$h_{kv}=h_q$，所以 $d_{kv}=d$
- GQA：$1<h_{kv}<h_q$
- MQA：$h_{kv}=1$

但 attention FLOPs 仍近似为：

$$
F_{\text{attn,prefill}}\approx4BL^2d
$$

$$
F_{\text{attn,decode}}\approx4BGTd
$$

原因是同一个 KV head 会被多个 query heads 共享。GQA/MQA 节省 KV 的生成和读取，并降低 KV cache 容量，但不会按照 $h_{kv}/h_q$ 的比例消除 query-key 点积。

## 4. MoE Transformer

MoE 必须区分：

- 总参数量 $N_{\text{total}}$
- 每个 token 实际使用的激活参数量 $N_{\text{active}}$

假设每层包含共享参数 $N_{\text{shared}}$、$E$ 个 experts、每个 expert 参数量 $N_{\text{expert}}$，并采用 top-k routing，则：

$$
\boxed{
N_{\text{active}}
\approx
N_{\text{shared}}+kN_{\text{expert}}
}
$$

而总参数量是：

$$
N_{\text{total}}
=
N_{\text{shared}}+EN_{\text{expert}}
$$

MoE prefill：

$$
\boxed{
F_{\text{MoE,prefill}}
\approx
2BLN_{\text{active}}
+F_{\text{attention}}
+F_{\text{router}}
}
$$

MoE decode：

$$
\boxed{
F_{\text{MoE,decode/token}}
\approx
2BN_{\text{active}}
+4nBTd
+F_{\text{router}}
+F_{\text{lm-head}}
}
$$

router 的计算量通常可近似为：

$$
F_{\text{router}}\approx2BLdE
$$

实际系统中还可以引入工程修正系数：

$$
\boxed{
F_{\text{actual}}
\approx
F_{\text{ideal}}\alpha_{\text{MoE}}
}
$$

$\alpha_{\text{MoE}}$ 用于吸收这些额外成本：

- expert capacity padding
- token duplication
- expert load imbalance
- dropped 或 padded tokens
- shared experts
- routing overhead

如果有 $s$ 个 shared experts，并且它们对所有 token 都执行：

$$
N_{\text{active}}
=
N_{\text{dense/shared}}+(s+k)N_{\text{expert}}
$$

FLOPs 只能描述算术量。MoE 的真实速度还可能受 all-to-all 通信、负载不均和小 GEMM 效率限制。

## 5. Sliding-window Attention

统一估算 sparse/window attention 最方便的方法，是定义每个 query 实际看到的 key 数。

若第 $i$ 个 query 可见的 key 数为 $s_i$：

$$
\boxed{
F_{\text{attn}}
\approx
4Bd\sum_i s_i
}
$$

### 5.1 Sliding-window Prefill

若每个 token 最多看当前及之前约 $W$ 个 token：

$$
\sum_{i=1}^{L}\min(i,W)
=
\begin{cases}
L(L+1)/2, & L\le W \\
W(W+1)/2+(L-W)W, & L>W
\end{cases}
$$

attention FLOPs：

$$
\boxed{
F_{\text{window,prefill}}
\approx
4Bnd\sum_{i=1}^{L}\min(i,W)
}
$$

当 $L\gg W$：

$$
\boxed{
F_{\text{window,prefill}}
\approx
4BnLdW
}
$$

总 prefill：

$$
\boxed{
F_{\text{prefill}}
\approx
2BLN_{\text{active}}
+4BnLdW
}
$$

### 5.2 Sliding-window Decode

单 token decode 只读取最近的 $\min(T,W)$ 个 KV：

$$
\boxed{
F_{\text{decode/token}}
\approx
2BN_{\text{active}}
+4Bnd\min(T,W)
+F_{\text{lm-head}}
}
$$

当 $T>W$ 后，attention FLOPs 不再随完整历史长度继续增长。

## 6. 一般 Block-sparse / Global-local Attention

如果每个 query 平均看到：

$$
S_{\text{eff}}
=W+G_{\text{global}}+\rho T
$$

其中：

- $W$：局部窗口
- $G_{\text{global}}$：全局 token 数
- $\rho T$：随机或结构化 sparse blocks 覆盖的历史 token 数

则 decode：

$$
\boxed{
F_{\text{sparse,decode/token}}
\approx
2BN_{\text{active}}
+4BndS_{\text{eff}}
+F_{\text{lm-head}}
}
$$

prefill：

$$
\boxed{
F_{\text{sparse,prefill}}
\approx
2BLN_{\text{active}}
+4BnLdS_{\text{eff,avg}}
}
$$

若只知道稀疏密度 $\rho$：

$$
F_{\text{attn,prefill}}
\approx
4Bn\rho L^2d
$$

需要注意，理论 sparse FLOPs 不等于真实加速。如果实现先计算完整 attention matrix 再应用 mask，实际计算仍接近 dense 的 $L^2$。只有真正跳过无效块的 sparse kernel 或 block-sparse kernel，才会兑现 FLOPs 的下降。

## 7. Linear Attention

Linear attention 的核心变化，是把类似下面的计算：

$$
QK^\top V
$$

重排为：

$$
Q(K^\top V)
$$

或者在 decode 时维护一个固定大小的 recurrent state。

设 feature/state dimension 为 $r$，value dimension 近似为 head dimension，则跨所有 heads 的主要状态更新和查询成本大致为 $O(BLdr)$。

prefill 的 attention 主项可以近似为：

$$
\boxed{
F_{\text{linear-attn,prefill}}
\approx
4BnLdr
}
$$

总 prefill：

$$
\boxed{
F_{\text{prefill}}
\approx
2BLN_{\text{active}}
+4BnLdr
}
$$

若 decode 维护 recurrent state，则每个新 token：

$$
\boxed{
F_{\text{decode/token}}
\approx
2BN_{\text{active}}
+4Bndr
+F_{\text{lm-head}}
}
$$

此时 attention 部分不再随历史长度 $T$ 增长。

不过 linear attention 家族差异很大：

- 核映射 $\phi(Q)$、$\phi(K)$ 有额外成本。
- Gated linear attention 还有 gate projection。
- Retention、RWKV、Mamba 类模型的状态更新公式并不完全相同。
- 某些模型训练时采用并行路径，推理时采用递归路径，两种路径的常数项不同。

因此，$4nLdr$ 适合作为状态交互主项的粗略估算，不应视为所有 linear attention 架构共享的精确常数。

## 8. 统一速查表

下表只展示主要缩放关系。完整公式还需要乘 batch size $B$ 和层数 $n$，参数矩阵部分的 $N_{\text{active}}$ 已按全模型统计。

| 架构 | Prefill attention | 单 token decode attention | 参数矩阵部分 |
| --- | ---: | ---: | ---: |
| Dense causal attention，按完整矩阵执行口径 | $\approx4nL^2d$ | $\approx4nTd$ | $2LN_{\text{active}}$ / $2N_{\text{active}}$ |
| Dense causal attention，按有效下三角口径 | $\approx2nL^2d$ | $\approx4nTd$ | 同上 |
| Sliding window | $\approx4nLWd$ | $\approx4n\min(T,W)d$ | 同上 |
| $\rho$-density sparse | $\approx4n\rho L^2d$ | $\approx4n\rho Td$ | 同上 |
| Global + local | $\approx4nL(W+G_{\text{global}})d$ | $\approx4n(W+G_{\text{global}})d$ | 同上 |
| Linear attention | $\approx4nLdr$ | $\approx4ndr$ | 同上 |
| MoE + 任意 attention | 使用对应 attention 项 | 使用对应 attention 项 | $N_{\text{total}}\rightarrow N_{\text{active}}$ |

Dense causal prefill 常见两种统计口径：

- 算法有效 FLOPs：只计算下三角，约为 $2nL^2d$。
- 实际 kernel/硬件工作量口径：如果按照完整 $QK^\top$ 和 $PV$ 统计，约为 $4nL^2d$。

进行硬件吞吐或 MFU 对比时，应采用实现真正执行和 profiler 使用的口径；进行算法复杂度比较时，可以使用 causal 有效 FLOPs，但需要明确标注。

## 9. 最推荐的通用公式

将各种架构统一起来：

$$
\boxed{
F_{\text{prefill}}
\approx
2BLN_{\text{active}}
+4Bnd\sum_{i=1}^{L}s_i
+F_{\text{extra}}
+F_{\text{lm-head}}
}
$$

其中，$s_i$ 是第 $i$ 个 query 实际参与计算的 key 数。

decode：

$$
\boxed{
F_{\text{decode-step}}
\approx
2BGN_{\text{active}}
+4Bnd\sum_{j=1}^{G}s_j
+F_{\text{extra}}
+F_{\text{lm-head}}
}
$$

不同架构只需要替换相应变量：

- Dense attention：$s_i\approx i$；如果 kernel 实际执行完整方阵，则使用对应执行口径。
- Sliding window：$s_i=\min(i,W)$。
- Sparse attention：$s_i$ 等于实际参与计算的 key 数。
- Linear attention：不用 $s_i$，改用固定 feature/state dimension $r$。
- MoE：将总参数量替换为每个 token 的激活参数量。

一句话版本：

$$
\boxed{
\text{FLOPs}
\approx
2\times\text{激活参数量}\times\text{本轮 token 数}
+4\times\text{层数}\times\text{query 数}\times\text{可见 key 数}\times d
}
$$

MoE 改“激活参数量”，稀疏 attention 改“可见 key 数”，linear attention 则把历史 key 数替换成固定状态或 feature dimension。

## 10. 7B Dense 模型的数量级示例

假设：

- $N_{\text{active}}=7\text{B}$
- $n=32$
- $d=4096$
- $L=T=4096$
- $B=1$
- 暂时忽略 LM head

Prefill 参数项：

$$
2NL
=2\times7\times10^9\times4096
\approx57.3\ \text{TFLOPs}
$$

Dense attention：

$$
4nL^2d
=4\times32\times4096^2\times4096
\approx8.8\ \text{TFLOPs}
$$

合计：

$$
F_{\text{prefill}}\approx66.1\ \text{TFLOPs}
$$

单 token decode 的参数项：

$$
2N\approx14\ \text{GFLOPs}
$$

attention 项：

$$
4nTd
=4\times32\times4096\times4096
\approx2.15\ \text{GFLOPs}
$$

合计约：

$$
16.2\ \text{GFLOPs/token}
$$

这说明：

- 在 4K 上下文下，7B 模型从纯 FLOPs 看仍主要由参数 GEMM 主导。
- Decode 的 wall time 未必由 FLOPs 主导，因为每一步都要读取权重和 KV cache，通常更偏 memory-bandwidth bound。
- 随着上下文继续增长，历史 KV attention 的占比会逐渐增加。

参数项与 attention 项的交叉点约为：

$$
2N\approx4nTd
$$

即：

$$
\boxed{
T_{\text{cross}}
\approx
\frac{N}{2nd}
}
$$

对上述 7B 模型：

$$
T_{\text{cross}}
\approx
\frac{7\times10^9}{2\times32\times4096}
\approx26.7\text{K}
$$

也就是说，从纯 FLOPs 看，约 27K 上下文时，单 token decode 的历史 attention 计算才接近参数矩阵计算。

## 11. 常见误区

1. $2N_{\text{params}}L$ 不包含 attention 的 $L^2$ 项。

2. MoE 不能直接使用总参数量，应使用每个 token 的激活参数量。

3. GQA/MQA 主要减少 KV projection、KV cache 和内存带宽，attention FLOPs 不会按照 KV head 数同比下降。

4. Decode 不能写成 $2N_{\text{params}}T$。有 KV cache 时，历史 token 不会重新运行整个模型。

5. 权重共享减少参数量，不一定减少实际计算量。Embedding 与 LM head tied 后，vocab projection 仍然需要执行。

6. 理论稀疏 mask 不保证真实稀疏计算。是否加速取决于 kernel 是否真正跳过被 mask 的块。

7. FLOPs 不等于延迟：

   - Prefill 往往更偏 compute-bound。
   - Decode 往往更偏 memory-bandwidth-bound。
   - MoE 可能受 all-to-all 通信限制。
   - Sparse/linear attention 可能受 kernel 常数、数据布局转换和小矩阵效率限制。

8. 不同 profiler 的 FLOPs 口径可能不同。有的把一次 FMA 记作 2 FLOPs，有的记作 1 op；比较前应统一口径。

