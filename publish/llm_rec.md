## 参考LLM建模方式

| 论文 | 公司 | 关键词 | 做法 |
|------|-----|------|----------------| 
|[TIGER](https://arxiv.org/pdf/2305.05065.pdf) | google | semantic_id粒度的生成式 | - 基于RQ-VAE聚类 <br> - 输入semantic_id list，预测下一个semantic_id <br> |
|[HSTU](https://arxiv.org/pdf/2402.17152.pdf) | meta | item_id粒度的生成式 | - 性别年龄等profile特征、action type也可以作为特殊的item_id <br> - 输入item_id list，预测下一个item_id <br> - 一种范式可以同时支持召回和排序 |
|[COBRA](https://arxiv.org/abs/2503.02453) | 百度 | semantic_id粒度的生成式 | - 和TIGER类似，只是给RQ-VAE再加一个由bert的CLS得到的emb <br> - 在线infer更复杂 <br> |
|[HeteroRec](https://arxiv.org/pdf/2503.01469) | 阿里 | 多模态+id生成 | - img/txt/id一起输入 <br> - listwise multi-step prediction <br> |

## 基于开源LLM

+ FFT：full finetuning
+ PT：prompt tuning
+ LAT：layerwise adapter tuning
+ OT：option tuning
+ T-FEW：few-shot peft

看着落地的

| 论文 | 公司 | 关键词 | 做法 | ab收益| tune方式|
|------|-----|------|----------------| -----|------|
|[KAR](https://arxiv.org/pdf/2306.10933) | 华为 | item llm+user llm | - 让LLM总结item得到item emb；<br> - 让LLM总结user历史得到user emb <br> - 两个emb过一个mmoe做融合得到新的两个emb，给推荐模型用 | 音乐推荐 涨了播放量 | frozen|
|[BAHE](https://arxiv.org/pdf/2403.19347) | 蚂蚁 | 预先计算原子用户行为 | - LLMs的预训练浅层提取来原子用户行为的emb，并存进离线db <br> - 从db里查出来，和item一起过LLMs的更深层可训练层  | 广告ctr+cpm | FFT上层LLM|
|[LEARN](https://arxiv.org/pdf/2405.03988) | 快手 | ItemLLM+user decoder| - item LLM固定，输入item特征得到item emb；<br> - 输入item emb过user的12层trans算dense all-action loss，<br> - 线上推荐模型里加user emb和item emb | 广告cvr+收入 | frozen|
|[BEQUE](https://arxiv.org/pdf/2311.03758)|阿里|SFT+离线模拟+PRO|query重写任务，SFT得到一个LLM，将其预测的若干个候选rewrites通过offline system的feedback得到排序，再通过PRO算法再tune LLM|电商搜索，gmv+单量|FFT|


看着没落地的


| 论文 | 公司 | 关键词 | 做法 | tune方式|
|------|-----|------|---------------------|------|
|[SLIM](https://arxiv.org/pdf/2403.04260) | 蚂蚁 | 蒸馏推荐理由| - 输入用户行为历史，大LLM(gpt)产出的推荐理由；<br> - 小llm(llama2)去蒸馏这个理由拿小llm去给出全量user的推荐理由， <br> - 通过BERT得到emb，给推荐模型用 | FFT |
|[DLLM2Rec](https://arxiv.org/pdf/2405.00338v1) | OPPO | 蒸馏推荐理由| 在SLMI的基础上设计了ranking蒸馏和embed蒸馏 | FFT |
|[LLM-CF](https://arxiv.org/pdf/2403.17688) | 快手 | 基于CoT数据集做RAG| - 拿推荐数据对llama2做sft，再用CoT的prompt让llama2对user+item+label产出一个推理过程，并通过bge得到emb，构建一个CoT数据集。 <br>- 在线拿当前用户+item的特征从这个数据集里ann出k个cot example的emb，和其他特征一起输入一个decoder，输出给推荐模型的sharebottom，额外加了一个CoT emb的重建loss | FFT|
|[ILM](https://arxiv.org/pdf/2406.02844)| google | 2阶段训练+q-former | - phase1：表示学习，交替训练两类表示学习（item-text表示学习，item-item表示学习）<br> - phase2：item-language model训练 | frozen|
|[EmbSum](https://www.arxiv.org/pdf/2405.11441)| meta | LLM摘要+t5 encoder | - 行为历史丢给LLM产出摘要，对应的hidden states给decoder自回归;<br> - 历史item过t5 encoder并concat过poly；<br>- item过t5 encoder过poly；| frozen|
|[Agent4Ranking](https://arxiv.org/pdf/2312.15450)|百度|agent rewrite+ bert ranking|query重写任务，多个人群当成多个agent，每个通过多轮对话产出一个rewrite，再合在一起经过bert+mmoe计算robust损失+accuracy损失。|frozen|

纯学术界

| 论文 | 关键词 | 做法 |tune方式 |
|------|------|---------------------|------|
|[CUP](https://arxiv.org/pdf/2311.01314)| LLM总结+bert双塔| 把用户的一堆历史评论扔给chatgpt，让它总结出128个token，然后丢给双塔bert，另一个塔是item的描述，freeze bert底层，只tune上层|last layer FT |
|[LLaMA-E](https://arxiv.org/pdf/2308.04913)| gpt扩展instruct| instruction formulating为写300个种子指令，让gpt作为teacher，对300个种子指令进行扩展，并由领域专家评估后，去重并保证质量，得到120k个指令作为训练集，再用lora去instruct tuning|lora|
|[EcomGPT](https://arxiv.org/pdf/2308.06966v1)| 一系列电商任务FFT BLOOMZ | 设置一系列的task(100多个task)来finetune BLOOMZ，包括命名实体识别、描述生成、对话intent提取等|FFT|
|[Llama4rec](https://arxiv.org/pdf/2401.13870) | prompt增强+数据增强，finetune | - prompt增强：在prompt里引入推荐模型的信息；<br> - 数据增强：通过LLM给推荐模型增加样本 <br> - adaptive aggregation：llm和推荐模型各自打分并用融合公式融合| FFT |
|[SAGCN](https://arxiv.org/pdf/2312.16275) | 分aspect打标、构图+gcn | - LLM为用户评论打标，确定aspect；<br> - 分aspect构建u-i图，并gcn| frozen|
|[GReaT](https://arxiv.org/pdf/2210.06280) | 表格型数据+LLM | 随机交换属性生成数据，finetune LLM预测属性 | FFT|
|[ONCE](https://arxiv.org/pdf/2305.06566)| 闭源LLM总结、开源LLM做encoder，u-i学ctr |闭源LLM输出文本（user profiler、content summarizer、personalized content generator），给开源LLM得到user表示，item过开源LLM得到item表示，二者内积学ctr| lora训开源，frozen闭源|
|[Agent4Rec](https://arxiv.org/pdf/2310.10108.pdf)|多智能体系统模拟交互，产出推荐样本| 先训一个推荐模型，然后构建一个多智能体系统，模拟和这个推荐模型交互，产出新的样本给推荐模型做数据增强 |仅训推荐模型，LLM frozen|
|[RecPrompt](https://arxiv.org/pdf/2312.10463)|两个LLM迭代出最佳prompt|给一个初始prompt，让LLM1得到推荐结果，拿一个monitor衡量这个结果和ground truth的mrr/ndcg，再用另一个LLM产出更好的prompt给第一个LLM用，如此迭代，得到一个best prompt|frozen|
|[PO4ISR](https://arxiv.org/pdf/2312.07552)|反思原因并refine/augment地迭代出最优的prompt|给初始prompt，收集error case让模型反思原因并refine出新的prompt，再augment出另一个prompt，并UCB选出最好的prompt，如此迭代|frozen|
|[TransRec](https://arxiv.org/pdf/2310.06491) | 受限生成 | - 将一个item表示成3部分：id+title+attr，设计三种对应的instruct-tuning任务；<br> - 引入一个特殊的数据结构（FM-index），并进行constrained beam search，让模型能生成候选集中的id/title/attr，<br> 再遍历全库候选，看不同facet的相似度（会考虑高热打压），加权融合出一个排序| lora|
|[E4SRec](https://arxiv.org/pdf/2312.02443)| 推荐id emb输入LLM | 推荐的id emb、prompt的emb一起输入LLM，最后一个词映射回推荐id emb的dim，去softmax | lora |


## 其他套路

工业界

| 论文 | 公司 | 关键词 | 做法 |
|------|-----|------|----------------| 
|[ExFM](https://arxiv.org/abs/2502.17494) | Meta | 两阶段蒸馏 | - 先训好teacher，并利用等待时间窗口为student数据集进行预估  <br> - 加了一些蒸馏loss |


学术界

| 论文 | 关键词 | 做法 |
|------|------|---------------------|
|[SLMRec](https://openreview.net/pdf?id=G4wARwjF8M) | 一阶段蒸馏 | teacher和student都拆成多个block，每个block间蒸馏 |
