[Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs](https://arxiv.org/pdf/2504.17432)

# 阶段一：文本判别知识蒸馏

![](../assets/unime-text-distill.png)

用纯文本数据来增强MLLM中LLM的嵌入能力，用了273k个句子对。

+ teacher：基于LLM的嵌入模型NV-Embed V2（对比训练中移除了causal mask）离线批量产出文本的emb
+ student：MLLM中把LLM部分剥离出来，输入prompt ```Summary the above sentences in one word: \n”```，最后一个token的emb当成输出
+ 蒸馏：teacher和student间算KL散度，LoRA训练

推理时：

+ 单模态输入：通过prompt的设置，决定只走对应的vision/text encoder
+ 图片+文字输入：通过prompt，各自过模型，然后把输出的2个emb进行融合 

# 阶段二：困难负样本增强指令微调

![](../assets/unime-img.png)

拿多模态的样本对来提升图文之间的对齐，用了662k个pair对：

+ query: img+prompt
+ doc: img

样本处理：

+ 干掉false negative：有些负样本和query的相似度太高了，干掉
+ 增加hard negative：除了正例和前面的false negative外，在负样本里找出k个和query最像的，当成hard negative

loss是infoNCE，然后用QLoRA来微调
