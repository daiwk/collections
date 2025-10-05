[EmbeddingGemma: Powerful and Lightweight Text Representations](https://arxiv.org/abs/2509.20354)

[https://huggingface.co/google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)

模型结构：

+ 把gemma这种decoder only的模型改成一个encoder-decoder架构(用如上图的T5-gemma的方法，[Encoder-decoder gemma: Improving the quality-efficiency trade-off via adaptation](https://arxiv.org/abs/2504.06225))。
+ 其实encoder部分和原模型是一样的，只是把mask改成双向的。在得到T个token的输出后，mean pooling（这里对比过更复杂的pooling方式，发现mean的效果最好）成一个768的emb。再过2层nn，最终映射到768。
+ 模型输入是task描述+原文本，例如检索任务里，query侧是"task: search result | query: {content}"，doc侧是"title: {title | 'none'} | text: {content}"

3个loss：

+ 基于inbatch负例的NCE：

$$
\mathcal{L}_C=\frac{1}{B} \sum_{i=1}^B\left[-\log \frac{e^{\operatorname{sim}\left(\mathbf{q}_i, \mathbf{p}_i^{+}\right) / \tau}}{w_i e^{\operatorname{sim}\left(\mathbf{q}_i, \mathbf{p}_i^{-}\right) / \tau}+\sum_{j=1}^B \mathbb{1}_{\mathrm{TN}}(i, j) e^{\operatorname{sim}\left(\mathbf{q}_i, \mathbf{p}_j^{+}\right) / \tau}}\right]
$$

其中的sim是cosine相似度，而$\mathbb{1}_{\mathrm{TN}}$把重复样本里的false negatives mask掉(其实就是保留batch内其他query的正的doc，还有当前query的其他正的doc)。另外，$w_i=\exp \left(\alpha \operatorname{sg}\left(\operatorname{sim}\left(\mathbf{q}_i, \mathbf{p}_i^{-}\right)\right)\right)$表示一个hard negative的难度，$\alpha$是个超参，设成5

$$
\mathbb{1}_{\mathrm{TN}}(i, j)= \begin{cases}0 & \text { if } q_i=q_j \text { or } p_i^{+}=p_j^{+} \\ 1 & \text { otherwise } .\end{cases}
$$

+ global orthogonal regularizer(GOR)的loss：

来自[Learning spread-out local feature descriptors](https://arxiv.org/abs/1708.06320)，可以让emb在emb空间内尽量展开，一方面对量化更鲁棒，另一方面也更方便ann检索。其实就是让随机挑选的一个pair对在统计意义上接近于从单位球面上的均匀采样，论文中衡量的是mean和二阶矩(second moment)，这里只用了二阶矩，因为发现二阶矩接近的时候，mean自然也会更接近。如下，对query间和正样本间均做了二阶矩的约束。

$$
\mathcal{L}_{\mathcal{S}}=\frac{1}{B(B-1)} \sum_{i, j \in \mathcal{B}: i \neq j}\left(\mathbf{q}_i^{\top} \mathbf{q}_j\right)^2+\frac{1}{B(B-1)} \sum_{i, j \in \mathcal{B}: i \neq j}\left(\mathbf{p}_i^{+\top} \mathbf{p}_j^{+}\right)^2
$$

+ embed distill loss

来自[EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval](https://arxiv.org/abs/2301.12005)，看着好像就是teacher和student的emb算一下L2距离，对query间、正样本间、hard负样本间均算这个loss：$\mathcal{L}_{\mathcal{D}}=\mathcal{L}_{\mathcal{D}}^{\mathrm{Q}}+\mathcal{L}_{\mathcal{D}}^{\mathrm{P}^{+}}+\mathcal{L}_{\mathcal{D}}^{\mathrm{P}^{-}}$

此外，NCE和GOR的loss也通过MRL作用到了各个子emb(512/256/128)上

训练的trick：

+ Encoder-Decoder Training：用如上的T5Gemma方式改成enc-dec结构，然后在gemma3的预训练数据上通过UL2（[Ul2: Unifying language learning paradigms](https://arxiv.org/abs/2205.05131)）进行预训练，然后拿这个encoder来用
+ Pre-finetuning：用大量的无监督语料训练，加大batch size以充足的emb负例，并让梯度更稳定。任务覆盖了问答、句子相似度、代码检索、网络搜索等。
+ Finetuning：高质量的数据，小的batchsize，加了hard negative，让batch内的样本更难。将任务分成三类：任务多样性（task diversity）、语言多样性（language diversity）、编码能力（coding capability），拿gemeni embedding工作里通过grid search找到的混合比例作为贝叶斯优化的种子，还从Dirichlet分布中随机采样10组不同的混合比例，形成多个版本的数据集（即不同任务侧重不同）。
+ Model Souping：用这些数据集分别训练模型，最后再把这些模型的权重平均得到最终的模型。结果就是最终的模型泛化能力更强，在各个任务上都比单个模型表现更好。
+ Quantization-Aware Training：3种类型的权重表示: int4 per-block, int8 per-block, and mixed-precision per-channel，用QAT训练
	
效果：

+ 在MTEB的多语言、英语和代码榜单上，是5亿参数以下模型里的第一名，而且是断层领先，性能甚至能和一些接近它两倍大的模型相抗衡。
+ 性能在权重被量化到int4/int8，或者嵌入维度被砍到128维时，依然保持得很稳定
