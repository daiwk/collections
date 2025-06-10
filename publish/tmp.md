# Qwen3-emb

[Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/pdf/2506.05176)

## 架构

![](../assets/qwen3-emb-arch.png)

+ emb模型：
    + query：输入instruction+query，拿eos对应的emb作为输出
    + doc：不用instrunction，直接输入doc，拿eos对应的emb作为输出
+ rerank模型：输入instruction+query和doc，拿如下prompt的输出经过lm head输出yes/no的概率（是否匹配）

```
<|im_start|>system
Judge whether the Document meets the requirements 
based on the Query and the Instruct provided. 
Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n
```

## 合成数据

数据：多样性，包括多语言、跨语言、多领域、多任务、长/短文本、代码检索

数据来源：

 + 人工标注数据：搜索(MS MARCO)、qa、MNLI、SNLI等，大概10M
 + 公开领域的弱监督数据：维基百科、redit、arxiv等，大概10亿，放弃了
 + 大模型合成数据：用Qwen3为每个doc生成相关问题，理论上可以构建无限数据

![](../assets/qwen3-emb-synthetic-data.jpg)

+ 根据文档，从一个比较大的角色库（比如大学生、小学生、公务员、商人等）里检索出若干个角色，用llm判断哪些角色适合这个文档，以及他们会问什么类型的问题（keyword/summary/逻辑推理），难度（大学/小学）、长度、语言，生成一个config
+ 拿角色+文档+config再给llm，生成真实的用户问题
+ 再用一个预训练好的emb模型先进行检索，过滤掉那些比较低质的数据（例如相似度太低或者根本检索不到相关文档）

## 训练

![](../assets/qwen3-emb-train.png)

多阶段训练

+ stage1：Large-Scale Synthetic Data-Driven Weak Supervision Training：
+ stage2：High-Quality Synthetic Data Utilization in Supervised Fine Tuning: 从合成数据中筛选出高质量数据，以及人工标注的高质量数据
+ stage3：Model Merging: 对stage2中的多个ckpt进行merge

rerank模型里stage1没啥用，直接用stage2和3，loss是sft的loss，即对yes/no去算交叉熵

embedding模型是3阶段，用一个改进的对比学习来训：

$$
L_{\text {embedding }}=-\frac{1}{N} \sum_i^N \log \frac{e^{\left(s\left(q_i, d_i^{+}\right) / \tau\right)}}{Z_i}
$$

+ 负例：$K$个hard neg $d^{-}_{i,k}$、batch内的其他query $q_j$、batch内的除了正负例的其他doc $d_j$
+ 计算负样本与query的相似度s1，正样本与query的相似度s2，如果s1比s2大超过一个阈值那就是false negative，即下面的$m_{ij}$

$$
Z_i=e^{\left(s\left(q_i, d_i^{+}\right) / \tau\right)}+\sum_k^K m_{i k} e^{\left(s\left(q_i, d_{i, k}^{-}\right) / \tau\right)}+\sum_{j \neq i} m_{i j} e^{\left(s\left(q_i, q_j\right) / \tau\right)}+\sum_{j \neq i} m_{i j} e^{\left(s\left(d_i^{+}, d_j\right) / \tau\right)}
$$

其中，

$$
m_{i j}= \begin{cases}0 & \text { if } s_{i j}>s\left(q_i, d_i^{+}\right)+0.1 \text { or } d_j==d_i^{+} \\ 1 & \text { otherwise, }\end{cases}
$$

训练时的策略：

+ 动态batchsize：训练数据长度不同，能接收的最大batchsize不同，所以对于不同长度用的bs不一样
+ gradient checkpointing：将大的bs切成小的sub-bs，先计算每个sub-bs的emb，因为gradient checkpointing不保存梯度，所以这个时候可以merge，merge完后再计算梯度
+ 训练的时候带了MRL loss

模型融合原理：多个ckpt，可能分别擅长不同的task，通过对模型参数的线性/球面插值，可以merge出一个对各任务都不错的模型。

参考[Improving general text embedding model: Tackling task conflict and data imbalance through model merging](https://arxiv.org/abs/2410.15035)的slerp，基于球面插值的融合方法，给定2个ckpt，分别算一下和训练前ckpt的差值，称为任务向量。然后拿这2个任务向量和原来ckpt的点构成的平面，使得少量训练数据和一个loss，算出合适的夹角和模长得到新向量，再去和其他ckpt重复这个步骤进行merge。

训练的时候用的lora，emb和rerank的hard negative设计不一样
