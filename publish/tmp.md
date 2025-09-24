
+ Thinker：输入文本、图像和音频，生成文本
+ Talker用于生成streaming speech tokens，输入是Thinker里图像和音频的高层语义表征，还有原始文本，这里不像Qwen2.5-Omni一样用Thinker的文本表征，可以更方便地给Talker嵌入其他外部模块（RAG/函数调用/安全过滤）方便人工干预，还可以为Thinker和Talker设置不同的prompt，例如分别控制Thinkier的风格和Talker的风格
+ 为了极致地降低延时，自回归地预测多码本序列：在每一步解码中，MTP模块输出当前帧的残差码本，随后Code2Wav（轻量级的ConvNet）增量地合成对应波形，实现逐帧流式生成。
+ 音频编码器：训了一个AuT(audio transformer)，attention-based encoder-decoder，基于2000万小时的音频数据(80%的中英文ASR数据，10%的其他语言ASR数据，10%的语音理解数据)训练。在attention layer之前，用Conv2D对filter bank特征进行8倍的下采样，将token rate降低到12.5Hz。用带了动态窗口的flash attention，来平衡实时的prefill caching效率和离线语音任务的效果。拿AuT的encoder来作为音频编码器，大概0.6B
+ 文本处理：用Qwen的tokenizer，词表大小151k
+ 视觉编码器：Qwen3-VL，543M参数，用SigLIP2-So400m初始化，在图片和视频的语料上训练
+ 位置编码：TM-RoPE（Time-aligned Multimodal RoPE），在Multimodal RoPE的基础上引入了时间信息

xx

+ 预训练：早期的预训练就混合了单模态和多模态的语料，并且设置了很多不同的prompt
    + Encoder Alignment Stage（S1）：LLM部分用Qwen3初始化，vision encoder用Qwen3-VL初始化，audio encoder用AuT初始化。之前的Qwen2.5-omni和Qwen2.5-VL是会fixed LLM，然后用图+文的数据去训练，同时在各encoder基础上有adapter去做模态对齐。这里把这个干掉了，而是这两个encoder只用单独自己模态的数据去训。
    + General Stage (S2)：在大约2T tokens的数据集（0.57T文本、0.77T语音、0.82T图像、0.05T视频、0.05T视频-语音）上训练
    + Long Context Stage (S3)：将最大的token length从8192扩展到32768，并把长语音和长视频的数据比例变大。
+ 后训练：
    + Thinker：有文本对话数据、图文对话数据、语音对话数据、混合模态对话数据，处理成ChatML格式
        + 轻量SFT：用不会偏离预训练分布太远的指令遵循的数据集来训
        + strong-to-weak蒸馏：teacher是Qwen3-32B或者Qwen3-235B-A22B，和Qwen3一样，先off-policy蒸馏（用teacher模型产出结果，做response蒸馏），再on-policy蒸馏（用student产出结果，并用于finetune，最小化student的logit和teacher的KL散度）
        + GSPO：用了两种reward：rule-based和model-based（LLM-as-a-judge，对于通用任务用Qwen3，视觉相关的任务用Qwen2.5-VL）
    + Talker：
        + 用数亿的带有多模态上下文的语音数据训练，建立一个从多模态表示到语音的单调映射
        + 用高质量数据进行CPT，减少幻觉并且提升生成语音的质量；同时加了long context训练，让模型能处理复杂输入并生成符合上下文的语音输出
        + 基于多语言语音数据构建偏好pair对，并用DPO来训练，提升多语言语音的生成能力
        + speaker finetuning，让模型能够采用特定的声音（应该是类似变声器那种）
    + Captioner：开源了Qwen3-Omni-30B-A3B-Captioner，对Qwen3-Omni-30B-A3B基于对大量detailed audio descriptions数据训练的模型，能够对任意语音输入产出详细的、低幻觉的caption。

