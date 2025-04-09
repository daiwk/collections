
# 2023–2025 大语言模型赋能推荐系统的研究与实践综述

随着大语言模型（LLM）的崛起，学术界和工业界开始探索将其应用于推荐系统，以利用LLM强大的语言理解、知识和生成能力来提升推荐效果。本文系统梳理了 **2023 年至 2025 年4月** 期间，arXiv上有关“大语言模型 + 推荐系统”的研究进展和落地实践，根据六大范式进行分类讨论，并列举每种范式下具有代表性的学术工作与工业应用案例。


## 1. Prompt式推荐（零样本/小样本/提示排序）

### 1.1 方法背景与技术框架  
Prompt式推荐是指通过**提示（prompt）**引导预训练的大语言模型直接执行推荐任务。在这种范式中，我们不对LLM进行专门的训练，而是通过精心设计的提示，将用户偏好、历史行为和候选商品等信息以自然语言格式提供给LLM，让其生成推荐结果。这一思路的核心在于利用LLM强大的**零样本/小样本学习**能力，让模型“即插即用”地充当推荐排序模型 ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=potential%20to%20approach%20recommendation%20tasks,can%20be%20biased%20by)) ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=recommendations%20in%20the%20zero,3%20to))。技术框架通常包括：将用户画像和上下文转换成提示模板，可能附加少量示例，然后让LLM推理输出用户可能感兴趣的项目或对候选项的偏好排序概率。Prompt式推荐的优势在于**无需针对推荐任务进行训练**即可利用LLM丰富的世界知识和语言推理能力，实现冷启动场景下的推荐 ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=preferences,even%20outperforming%20some%20strong%20sequential))。同时，LLM生成的推荐结果还天然带有解释性（因为LLM可以给出理由）。然而挑战也很明显：**候选集合极大时的效率问题**（LLM对长列表评分的开销极高）以及**提示设计**的合理性（如何确保LLM理解用户历史行为顺序、不受提示中位置偏见影响等) ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=template%20and%20conduct%20extensive%20experiments,available%20at%20this%20https%20URL))。因此，该范式的技术框架常辅以**提示工程**（优化prompt模板、加入链式思考等）和**候选精排**（先用传统模型筛选小规模候选，再交由LLM决策）等技术来提升效果和效率。

### 1.2 代表性研究工作  
- **Large Language Models are Zero-Shot Rankers for Recommender Systems (2023)** ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=template%20and%20conduct%20extensive%20experiments,available%20at%20this%20https%20URL))：Yupeng Hou 等提出将推荐视为**有条件的排序任务**，通过精心设计prompt模板，将用户的序列历史行为作为条件，将待排序的物品作为候选列表嵌入提示，令GPT-3/4这类LLM直接输出评分来进行排序。结果表明，在零样本设置下，LLM对推荐排序表现出有前景的能力，可逼近甚至挑战无训练的传统模型。他们还发现LLM提示排序存在易受**位置和流行度偏见**等问题，并通过特殊prompt和自举（bootstrapping）策略缓解偏差 ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=template%20and%20conduct%20extensive%20experiments,available%20at%20this%20https%20URL))。该工作代码已开源，验证了即使不针对推荐微调，LLM也可作为强力排序器。*(是否生成式：否，LLM用于评分排序；是否使用LLM结构：是，直接调用GPT模型；代码开源：是)*

- **Zero-Shot Next-Item Recommendation using Large PLMs (2023)** ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=preferences,even%20outperforming%20some%20strong%20sequential))：Lei Wang 和 Ee-Peng Lim 提出零样本下的大语言模型下一步推荐方法。他们设计了一个三阶段的Prompt方案“Next-Item Recommendation (NIR) Prompting”：首先利用外部模块根据用户历史筛选候选集合，然后提示GPT-3依次**总结用户兴趣 -> 回顾代表性历史物品 -> 生成推荐列表** ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=preferences,even%20outperforming%20some%20strong%20sequential))。在MovieLens数据上，GPT-3零样本推荐的命中率甚至超过了一些训练的深度序列模型 ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=filtering,to%20use%20LLMs%20as%20recommenders))。这证明了大模型强大的推理泛化能力。作者也开源了实现代码。*(是否生成式：部分，是让GPT-3生成物品名称列表；是否使用LLM结构：是，调用GPT-3 API；代码开源：是)*

- **RecPrompt: News Recommendation via Self-tuning Prompting (2024)** ([[2312.10463] RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models](https://arxiv.org/abs/2312.10463#:~:text=tasks,based%20explanations))：Dairui Liu 等研究了在新闻推荐中自动优化prompt。提出 **RecPrompt** 框架，由一个新闻推荐模型和一个提示优化器组成，采用**迭代自举**的方法自动调整Prompt。 ([[2312.10463] RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models](https://arxiv.org/abs/2312.10463#:~:text=tasks,based%20explanations)) 实验使用 GPT-4 对400名真实用户进行新闻推荐，自适应地调整提示后，点击率等指标相比SOTA深度模型有显著提升（AUC提升3.36%，nDCG@5提升9.64%等） ([[2312.10463] RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models](https://arxiv.org/abs/2312.10463#:~:text=tasks,based%20explanations))。此外引入TopicScore评估LLM总结用户兴趣主题的能力。RecPrompt是首个将**Prompt工程与推荐模型闭环结合**的工作。*(是否生成式：否，GPT-4用于排序评分；是否使用LLM结构：是，调用GPT-4 API；代码开源：未明确)*

- **LLM-Rec: Prompting LLMs for Text-based Recommendation (2023)** ([[2307.15780] LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/abs/2307.15780#:~:text=approach%2C%20coined%20LLM,techniques%20to%20boost%20the%20recommendation))：Hanjia Lyu 等聚焦**文本内容推荐**，利用LLM提升物品文本描述的表示质量。他们设计了四类prompt策略，让GPT生成更丰富的物品属性描述作为额外特征供一个简单的推荐模型（如MLP）使用。结果显示，经LLM扩充文本后，哪怕简单模型的效果也**媲美甚至优于**复杂的内容推荐模型 ([[2307.15780] LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/abs/2307.15780#:~:text=approach%2C%20coined%20LLM,techniques%20to%20boost%20the%20recommendation))。凸显了提示式输入增强在推荐中的价值。*(是否生成式：否，LLM生成辅助信息；是否使用LLM结构：是，调用GPT生成文本；代码开源：未明确)*

*（更多相关工作：例如 TALLRec ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=An%20Efficient%20All,2022%29))使用小规模示例提示结合LoRA微调提升LLM推荐效果，PALR提出个性化提示生成策略等。下文将在其他范式部分介绍。）*

### 1.3 学术界成果与方法进展  
在Prompt式推荐范式下，学术界近两年进行了大量探索，逐步揭示了LLM直接用于推荐的潜力与局限：

首先，一系列开创性工作验证了**LLM零样本推荐的可行性**。例如 Hou 等发现GPT-3.5/4在排序任务上具有**惊人的零样本能力**，只需将用户历史和候选项以适当格式嵌入提示，LLM即可给出符合偏好的排序结果 ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=can%20be%20alleviated%20using%20specially,available%20at%20this%20https%20URL))。Wang & Lim 的研究进一步证明，经过良好设计的提示，LLM推荐性能甚至可超越训练有素的深度模型 ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=filtering,to%20use%20LLMs%20as%20recommenders))。这些结果表明，大模型已经内隐学习到了与推荐相关的知识和推理能力，哪怕从未专门训练过推荐任务。

针对**提示设计**，学术界识别出一些关键挑战并提出对策。例如，LLM在顺序推荐中**可能忽视交互顺序**，对提示中最近的物品给予过高权重，或者受热门物品词频影响而产生偏置 ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=template%20and%20conduct%20extensive%20experiments,available%20at%20this%20https%20URL))。为此，研究者提出在提示中显式标注时间顺序，或加入例如“不要仅根据流行度推荐”等指令，来纠正LLM的认知偏差 ([[2305.08845] Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845#:~:text=template%20and%20conduct%20extensive%20experiments,available%20at%20this%20https%20URL))。还有工作引入**Chain-of-Thought**等技巧，引导模型逐步分析用户兴趣再给结论，提升推荐准确性和多样性。

另外，为了解决**候选集合过大**的问题，不少研究采用了**两阶段策略**：先用轻量模型进行召回或粗排，选出top-N候选，再让LLM在较小集合上精排。例如 Zero-Shot NIR方法就借助外部模块筛选电影候选，再让GPT-3在10部候选中输出排名 ([[2304.03153] Zero-Shot Next-Item Recommendation using Large Pretrained Language Models](https://ar5iv.org/pdf/2304.03153#:~:text=preferences,even%20outperforming%20some%20strong%20sequential))。这样既降低了LLM推理负担，又减少了LLM可能“凭空”生成无关项的风险。

自适应的Prompt优化也是进展热点。RecPrompt工作展示了**自动Prompt调整**可以进一步提升效果 ([[2312.10463] RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models](https://arxiv.org/abs/2312.10463#:~:text=tasks,based%20explanations))。不像人工提示，RecPrompt用反馈不断修正提示内容，使LLM更关注用户兴趣未被满足的部分，从而在几个自举回合后取得显著性能增益。这种**人机协同优化**理念为Prompt式推荐打开了新思路。

总体而言，学术研究已经**证实LLM可以作为零样本推荐模型**使用，在冷启动场景下表现出色，同时为应对其局限开发了多种prompt工程策略和辅助机制。在这些工作的推动下，Prompt式推荐正逐渐从实验走向更实际的问题规模，为工业界探索打下基础。

### 1.4 工业界落地案例  
在工业界，对Prompt式推荐的尝试也已经开始出现，一些大型互联网公司利用LLM的通用能力来简化推荐系统架构或改进冷启动表现：

- **LinkedIn（领英）— “360Brew” 通用排序模型**： ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=360Brew%C2%A0%C2%A0V1,teams%20of%20a%20similar%20or))领英团队在2024年提出了名为 *360Brew* 的超大规模基础模型（1500亿参数）用于个性化推荐和排序任务 ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=360Brew%C2%A0%C2%A0V1,teams%20of%20a%20similar%20or))。360Brew采用**文本接口**，将用户行为和候选内容全部“verbalize”成文本提示输入单一的Decoder-only LLM。一方面，一个模型即可统一处理领英站内30多个推荐与排序任务，实现了**多任务一体化** ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=approach%3A%20,DAGs%29%20of))；另一方面，由于利用了LLM的推理和知识泛化能力，该模型在无额外微调的前提下对新领域任务表现出零样本适应性 ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=textual%20interface%20due%20to%20their,researchers%20and%20engineers%20over%20a))。离线实验显示，360Brew在多个子任务上达到或超过了原有专门模型的性能 ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=trained%20and%20fine,larger%20size%20than%20our%20own))。这一工业级案例表明，通过精心设计Prompt接口和训练流程，大模型有潜力**替代传统繁杂的推荐系统流水线**。*(场景：职场社交Feed、求职推荐等多场景；融合方式：全Prompt输入；是否生成式：否，生成得分排序；是否复用LLM结构：是，自研大模型；A/B测试：尚在预生产，离线指标已与生产持平 ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=360Brew%C2%A0%C2%A0V1,teams%20of%20a%20similar%20or)))*

- **华为 Noah’s Ark Lab — LLMTreeRec 冷启动推荐**： ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=The%20lack%20of%20training%20data,item%20tree%20to%20improve%20the)) ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=performance%20under%20the%20system%20cold,Lab%2FLLMTreeRec))面对真实应用中新品/新用户缺乏交互数据的冷启动问题，华为提出并部署了 *LLMTreeRec* 框架 ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=The%20lack%20of%20training%20data,item%20tree%20to%20improve%20the))。其核心思想是将海量候选物品组织成多叉树结构，通过多轮Prompt引导LLM逐层**筛选**：每一层LLM根据用户偏好选择下一层的分支，逐步缩小候选集合，最终定位最符合用户兴趣的物品 ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=recommendations,in%20two%20widely%20used%20datasets))。这种树型检索大大提高了LLM决策的效率，使之可应用于上百万规模的候选集。离线结果表明，在**系统冷启动**场景下该方法效果达到SOTA水平，甚至接近有充足训练数据的深度模型 ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=LLMTreeRec%2C%20which%20structures%20all%20items,Learning))。更重要的是，LLMTreeRec已经在华为某工业推荐系统中上线，并通过在线A/B测试验证了优于既有模型的性能 ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=performance%20under%20the%20system%20cold,Lab%2FLLMTreeRec))。这是Prompt式LLM直接用于工业推荐决策的成功案例之一。*(推荐场景：新用户商品推荐；融合方式：Prompt引导逐层筛选树结构候选；是否生成式：否，生成下一步选择；是否复用LLM结构：是，调用开创性LLM；A/B测试：线上点击率优于基线模型 ([LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/html/2404.00702v3#:~:text=performance%20under%20the%20system%20cold,Lab%2FLLMTreeRec)))*

*(除上述案例外，一些公司也开始探索让内部通用大模型通过提示来执行推荐任务。例如，有报道指淘宝等电商在开发类似ChatGPT的购物助手，通过对话为用户推荐商品；又如B站据传尝试用大模型对冷门内容做长尾推荐。这些实践大多处于试验阶段，公开的细节和指标有限，因此未在此详述。)*

## 2. 特征与语义增强（内容理解、冷启动、知识注入）

### 2.1 方法背景与技术框架  
特征与语义增强范式侧重于**利用LLM丰富的语义理解能力**来改进推荐系统的输入特征表示和知识获取。传统的推荐系统很大程度上依赖**ID嵌入**和稀疏的历史行为，这往往忽略了物品内容和上下文中的大量语义信息 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=embedding%20to%20capture%20latent%20associations,item%20encoders%20and%20freezing%20LLM))。例如，一本书的文本描述、商品的属性和评论、用户发表的文本动态，这些都蕴含着潜在的偏好线索。但以往模型难以充分利用这些**非结构化文本**信息。LLM作为在海量语料上训练的模型，具备出色的**自然语言理解和常识推理**能力，因而被用于提升推荐系统对内容和语义的感知：一是**生成更优的特征表示**，如用GPT对商品描述做摘要提炼关键信息，缓解冷启动物品缺少历史互动的问题；二是**引入外部知识**，例如利用LLM从知识图谱或百科中抽取知识点，增强推荐的多样性和准确性；三是**判别噪声与真假**，利用LLM识别评论的真假、标签的相关性，从而清洗训练数据等。

在技术框架上，这一范式通常通过两种途径融入LLM：其一，**LLM作为特征提取器**，即冻结预训练LLM，用其编码物品描述、用户评论等文本，得到高质量的向量表示，与用户IDEmbedding结合输入推荐模型 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=comprehend%20and%20reason%20about%20textual,demonstrate%20the%20efficacy%20of%20our))；其二，**LLM作为知识发现工具**，通过prompt引导LLM去推理物品关联关系（如“买了A是否也可能买B”），将推理得到的新特征（例如A和B的互补关系）注入推荐模型的数据管道。总体而言，特征与语义增强范式将LLM视为现有推荐系统的“外挂大脑”，帮助传统模型看懂看透以前看不懂的内容，从而提升对用户兴趣的刻画，特别在冷启动和长尾内容场景下改善推荐质量。

### 2.2 代表性研究工作  
- **LLM-Rec: 利用LLM丰富物品文本描述 (2023)** ([[2307.15780] LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/abs/2307.15780#:~:text=approach%2C%20coined%20LLM,techniques%20to%20boost%20the%20recommendation))：该工作在第1节已提及，由于属于特征增强范式，这里归类说明。Lyu 等提出用GPT模型生成包含常识和细节的物品文本。例如对一个电影，提示GPT列出其主要情节、风格、类似影片等关键词，得到扩充描述，再将其作为额外特征并入推荐模型训练。实验发现，经LLM扩充后的文本特征大幅提高了推荐效果：在电影、商品等数据集上，一个简单的MLP模型使用增强文本即可超越复杂的内容推荐模型 ([[2307.15780] LLM-Rec: Personalized Recommendation via Prompting Large Language Models](https://arxiv.org/abs/2307.15780#:~:text=approach%2C%20coined%20LLM,techniques%20to%20boost%20the%20recommendation))。这证明了LLM生成的**语义特征**能有效弥补原始描述的信息不足。*(是否生成式：否，LLM生成中间特征；是否复用LLM结构：是（推理生成特征时调用）；代码开源：未明确)*

- **LEARN: 冻结LLM作为工业推荐的知识塔 (2024)** ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=comprehend%20and%20reason%20about%20textual,demonstrate%20the%20efficacy%20of%20our))：Jian Jia 等提出了一个面向工业应用的知识适配框架LEARN ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=comprehend%20and%20reason%20about%20textual,scale))。他们将预训练LLM（如GPT系列）的参数**冻结**，仅将其作为物品文本内容的编码器，以保持LLM对开放领域知识的记忆 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=knowlEdge%20Adaptive%20RecommeNdation%20,art%20performance))。同时设计一个协同过滤塔（IDembedding塔），通过双塔结构融合LLM的“开世界”语义知识与传统“封闭世界”协同知识。在大规模工业数据集上，LEARN相比纯ID模型有明显性能提升，并在六个Amazon商品评论数据集取得SOTA ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=To%20bridge%20the%20gap%20between,the%20superiority%20of%20our%20method))。更重要的是，作者报告了在真实线上业务上的A/B测试成功，表明该语义增强策略在工业环境中切实可行 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=practical%20industrial%20application,art%20performance))。他们也公开了代码以促进后续研究。*(是否生成式：否，将LLM用作编码器；是否复用LLM结构：是（参数冻结使用）；代码开源：是)*

- **Breaking the Barrier: 利用知识推理提升工业推荐 (2024)** ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=task,level%20of%20accuracy%20in%20the)) ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=improvement%20achieved%20by%20our%20method,world%20industrial%20recommendation%20scenarios))：这是一篇由蚂蚁集团提出的工业报告。作者关注电商推荐中的**互补品推荐**（买了A后推荐B）。传统模型难以捕捉面包和牛奶这种常识性的互补关系，该工作引入Claude 2 LLM，通过Prompt提供成对商品，让LLM判断二者是否存在互补购买关系，并给出理由 ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=task,level%20of%20accuracy%20in%20the)) ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=Description%20of%20the%20input%20data,of%20the%20purposes%20of%20the))。他们将LLM输出的判断结果构建为“互补知识图”，纳入推荐系统的召回和排序特征中。在线实验在支付宝的优惠券和商品推荐场景中进行：随机10%用户流量的对照试验显示，引入LLM推理知识的方案（LLM-KERec）相比原系统**点击和转化率显著提升**，如优惠券兑换量提升6.24%和10.07%，商品GMV提升6.45% ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=improvement%20achieved%20by%20our%20method,world%20industrial%20recommendation%20scenarios))。这种方法**无需额外数据**就让推荐系统具备了常识推理能力，是LLM知识注入在工业界的成功应用。*(推荐场景：支付宝优惠券和商品推荐；融合方式：LLM推理互补关系形成知识图特征；是否生成式：部分，生成“Yes/No”判断及解释；是否复用LLM结构：是，调用Claude API；A/B测试：转化提升6~10% ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=improvement%20achieved%20by%20our%20method,world%20industrial%20recommendation%20scenarios)))*

- **FilterLLM: 文本到用户分布的冷启动推荐 (2025)** ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=cold,start%20performance))：阿里巴巴团队提出利用LLM解决海量新物品的冷启动分发问题。他们训练了一个名为FilterLLM的模型：在LLM的输出空间扩展加入大量**用户ID的专属token**，并设计提示仅输入物品内容文本，促使LLM直接输出一个用户ID分布（即预测哪些用户可能喜欢该新物品） ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=,%E2%80%9D)) ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=))。为高效训练如此海量的新token，他们使用协同过滤的嵌入初始化技巧，结合对比学习来适配LLM ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=Introducing%20a%20user%20vocabulary%20into,for%20initializing%20user%20token%20embeddings)) ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=))。在线A/B测试进行了两个月，覆盖日均3亿PV的推荐场景。结果显示，FilterLLM相比之前的冷启动模型在**召回速度上提升一个数量级**，同时冷启动推荐效果也有提升 ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=cold,start%20performance))。这证明了LLM可以直接输出推荐所需的用户分布，大幅加速新内容触达用户的效率。*(推荐场景：电商新商品冷启动；融合方式：扩展LLM词表表示用户，输入物品文本输出用户ID概率分布；是否生成式：是，生成用户列表；是否复用LLM结构：是（微调LLM加新词）；A/B测试：线上速度提高数量级且效果提升 ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=cold,start%20performance)))*

*(其他相关工作：如ONCE ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=ONCE%C2%A0%28Liu%20et%C2%A0al,Similarly%2C%20LlamaRec%C2%A0%28Luo))首先尝试让LLM生成用户可能点击的新闻作为数据增强；LLM-EKF提出用LLM填充知识图谱中的缺失边以改进推荐召回；LARR通过LLM理解实时场景文本来辅助短视频推荐等等。)*

### 2.3 学术界成果与方法进展  
在特征与语义增强方面，学术界的研究丰富了推荐系统对内容和知识的利用，主要进展包括：

**1）内容理解与表征迁移：** 多项研究成功将预训练LLM用作**强大的文本编码器**嵌入到推荐模型中。例如，LEARN框架通过双塔网络将冻结的LLM文本编码与可训练的ID嵌入结合，显著提升了推荐性能 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=knowlEdge%20Adaptive%20RecommeNdation%20,art%20performance))。又如CoLLM等工作尝试将协同过滤产生的embedding直接融合进LLM的隐层，使LLM同时学习文本和协同信号 ([Large Language Models Are Universal Recommendation Learners](https://arxiv.org/html/2502.03041v1#:~:text=To%20address%20the%20limitations%20of,a%20good%20balance%20between%20the))。总体来看，这类方法证明了**预训练语言模型的知识可以迁移**到推荐领域，只需很少甚至不需要调整参数，就能提供比传统方法更通用、更语义丰富的表示，从而改善冷启动和跨领域场景下的效果。

**2）知识注入与推理增强：** 针对推荐系统面临的知识瓶颈（如互补关系、因果关系），研究者利用LLM的**推理能力**引入外部知识源。蚂蚁金服的工作展示了让LLM充当“知识判别器”，自动挖掘商品之间的语义关系能够极大拓展推荐系统的视野 ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=task,level%20of%20accuracy%20in%20the)) ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=Description%20of%20the%20input%20data,of%20the%20purposes%20of%20the))。还有学者利用LLM读取商品的维基百科信息，提取属性标签补充到物品特征中，或让LLM根据剧情简介推测用户对影片细分元素（演员、主题）的喜好，以增强个性化。实验普遍表明，补入这些LLM获取的知识后，推荐的准确率和多样性都有所提高，**尤其在长尾物品**（历史数据稀少）上效果提升更明显。

**3）软硬件权衡与高效部署：** 为了使LLM的语义增强能力在工业规模可用，学术界也在探索更高效的集成方案。冻结LLM参数是一种思路，可避免大模型在训练中“灾难性遗忘”其预训练知识 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=knowlEdge%20Adaptive%20RecommeNdation%20,art%20performance))。此外，一些方法通过知识蒸馏，将LLM提取的内容特征用小模型来近似，从而减少在线依赖。还有研究考虑将LLM放在推荐系统的上游离线阶段，用于**数据增强或清洗**——例如Liu等让LLM识别序列互动数据中的噪声并剔除 ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=utilize%20the%20LLM%20to%20judge,noisy%20samples%20in%20sequential%20recommendation))——这种方式也能降低实时开销。在学术界的努力下，LLM增强推荐正朝着**低成本、高收益**的方向发展。

总的来说，语义增强范式的研究充分证明：LLM可以赋予推荐系统“洞察力”和“常识”，不仅提升了推荐质量，也拓展了推荐系统可用的信息源，使其能够整合包括文本、知识图谱、跨域信息在内的多模态线索。

### 2.4 工业界落地案例  
工业界对LLM进行特征与语义增强的落地已有一些成功报道，说明这一范式在大规模应用中具有现实价值：

- **阿里巴巴：商品互补推荐中的知识推理** – 前文提及的蚂蚁集团案例 ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=improvement%20achieved%20by%20our%20method,world%20industrial%20recommendation%20scenarios))展示了LLM推理互补知识在电商业务中的价值。他们以极小的代价（仅调用LLM推理判断）显著提升了三大实际场景的核心指标 ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=improvement%20achieved%20by%20our%20method,world%20industrial%20recommendation%20scenarios))。该案例的成功在业界引起关注，证明利用LLM获取常识性知识可以**破解长尾商品推荐难题**，提升用户体验。目前，该方案已在支付宝的“超级福利”、支付结果页等场景上线部署。

- **某头部短视频平台：语义双塔匹配** – LEARN框架的作者来自业界（字节跳动/Kuaishou等可能背景），其在投稿中明确指出该方案经过了**真实工业数据集和在线测试** ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=To%20bridge%20the%20gap%20between,art%20performance))。虽然具体公司未指明，但从作者包含某互联网公司研究员且提及“工业应用”，我们推测LEARN或类似的LLM+双塔模型已经在短视频推荐或电商推荐中试验部署，通过对商品文本、视频描述等内容建模，提升了推荐召回的泛化能力。据论文描述，在线A/B测试结果验证了语义增强带来的效果增益 ([[2405.03988] LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988#:~:text=To%20bridge%20the%20gap%20between,art%20performance))。这表明工业界开始接受“冻结LLM当特征提取”这种新范式，将其视为改进大型推荐系统的一种可行且低风险的插件。

- **阿里巴巴：冷启动用户分发加速** – FilterLLM的方法已经在阿里内部进行了长周期的大流量测试 ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=cold,start%20performance))。在内容生态非常丰富的平台（如淘宝、优酷）中，新内容如何高效触达潜在喜好用户一直是难题。FilterLLM通过让LLM“一步到位”地给出新内容的目标用户列表，把过去需要多轮召回排序的流程大大简化。据报道，两个月的线上实验中，不仅新内容曝光速度提升了一个量级，用户的点击和停留等表现也优于对照组 ([FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/html/2502.16924v1#:~:text=cold,start%20performance))。据悉，阿里已在考虑将该方案用于实际业务中，以增强新商品、新视频的分发效率。

总之，工业界对特征与语义增强范式的实践表明：**LLM可以作为提升推荐系统效果的“催化剂”**，在不推翻原有架构的情况下，通过丰富特征和知识，取得令人瞩目的收益。随着更多公司验证LLM在内容理解和知识注入上的价值，可以预见这一范式将得到更广泛的应用。

## 3. 网络结构融合与参数高效微调（如 LoRA, MoE, Prompt Tuning 等）

### 3.1 方法背景与技术框架  
尽管直接使用LLM进行推荐（Prompt范式）和将LLM当工具辅助推荐（特征增强）都取得了成功，但还有一种思路是**更紧密地融合LLM与推荐模型的网络结构**，通过**参数微调**来让LLM更好地适配推荐任务。这一范式的出发点在于：预训练LLM并非专为推荐而生，其输出和训练目标与推荐目标存在差异。如果能在**不牺牲LLM通用能力**的前提下，对其进行适度微调，让模型更“懂”用户与物品的匹配关系，那么推荐性能有望进一步提升。然而，直接Fine-tune整个LLM代价高昂且可能过拟合小数据。为此，学术界引入了NLP领域成熟的**参数高效微调（PEFT）**技术，如 LoRA（低秩适配） ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=An%20Efficient%20All,2022%29))、Prefix/Prompt Tuning、Adapter等，将少量新参数插入或融合到LLM中，利用极低的开销来调整模型行为。这类方法通过**冻结大部分LLM参数，仅训练小规模参数模块**，既保持了LLM强大的语言知识，又学到了推荐领域的新模式。

网络结构融合方面，也有工作尝试结合**协同过滤结构**与LLM。例如，在LLM输出层接入一个用于生成推荐得分的专用头，或者增加一层用于处理ID Embedding的MoE专家网络，从而**融合显式ID信号**与LLM隐式语义。在技术框架上，这类范式通常需要**一定程度的模型训练**：要么是在已有LLM上附加新结构并训练（可能需要大规模交互数据做微调），要么是预训练时就考虑推荐任务目标（如通过多任务学习预训练一个同时具备语言和推荐能力的基础模型）。

简单来说，网络结构融合与高效微调范式追求**“1 + 1 > 2”**的效果：将LLM的语言天赋和传统模型的协同效应合二为一，通过小幅训练调整，让LLM成为更专业的推荐模型，同时保留其生成解释等额外能力。

### 3.2 代表性研究工作  
- **TALLRec: 基于LoRA的高效微调框架 (RecSys 2023)** ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=An%20Efficient%20All,2022%29))：Bao 等提出“TALLRec”，旨在**高效地对齐LLM与推荐任务**。他们采用LoRA方法对一个预训练语言模型的部分参数进行低秩适配微调，只训练不到1%的参数量，却成功让LLM学会了序列推荐的模式 ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=To%20bridge%20this%20gap%2C%20TALLRec,2022%29))。实验在MovieLens等数据集上表明，TALLRec在推荐准确率上优于不经微调直接用LLM的方法，也接近传统深度模型的水平，但训练成本大幅降低 ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=To%20bridge%20this%20gap%2C%20TALLRec,2022%29))。这一工作验证了PEFT技术在推荐领域的可行性。*(是否生成式：否，微调后模型用于预测评分；是否复用LLM结构：是（LoRA微调GPT）；代码开源：是)*

- **CoLLM: 协同Embedding融合LLM (2023)** ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=,Models%20with%20Sequential%20Recommenders%2C%20arxiv))：该工作探索将**协同过滤的嵌入直接融合进LLM**。作者将预训练好的用户和物品ID embedding引入LLM的隐藏层，使模型在生成下一词时同时考虑这些embedding的影响。这样LLM既保留语言模型对文本的理解，又能利用用户-物品交互的协同信号。实验表明，这种融合提高了推荐准确率。*(是否生成式：否；是否复用LLM结构：是（融合结构）；代码开源：未明确)*

- **Lifelong Personalized LoRA (2024)**：针对推荐场景动态变化，Chen 等提出让每个用户拥有一组LoRA参数，对LLM进行个性化微调并能持续更新 ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=%2A%20Review,Collaborative%20Large%20Language%20Model%20for))。这种架构相当于为LLM加装用户记忆模块，随着用户行为新增，不断调整小规模参数，实现**终身学习**，避免模型过时。离线评估显示对长期用户偏好捕获更准确。*(是否生成式：否；是否复用LLM结构：是（多个LoRA模块）；代码开源：未明确)*

- **Large Language Models meet Collaborative Filtering (KDD 2024)**：Wang 等提出一套**“全能型”LLM推荐系统**，他们在一个基础LLM上同时融入了用户和物品的多个交互视图，构建统一模型处理评分预测、点击率预估等任务 ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=,paper))。模型采用高效微调和蒸馏结合，使其在推荐数据上表现突出。*(是否生成式：否；是否复用LLM结构：是；代码开源：未明确)*

*(其他相关工作：如Prompt Tuning用于提取用户评论中的偏好要素、HLLM结构用于层次化地分解推荐意图、E4SRec提出端到端的LLM序列推荐方案等。)*

### 3.3 学术界成果与方法进展  
网络结构融合与参数微调范式的研究丰富了将LLM改造为推荐模型的思路，主要进展体现在以下几方面：

**1）参数高效微调显著提升效果：** 多项研究证明，仅需很小的训练代价即可让LLM的推荐性能飞跃式提高。TALLRec 的实验结果表明，通过LoRA微调，LLM对交互序列的建模能力明显增强，在Hit率等指标上大幅超越零样本LLM ([An Efficient All-round LLM-based Recommender System - arXiv](https://arxiv.org/html/2404.11343v1#:~:text=To%20bridge%20this%20gap%2C%20TALLRec,2022%29))。这说明预训练LLM中蕴含的很多知识可以被**微调唤醒**来服务推荐任务，而PEFT方法提供了一个经济高效的途径。相比微调整个模型，LoRA等方法不但计算开销低，也降低了过拟合风险，许多研究都报告PEFT微调后的LLM在小数据集上依然具有良好泛化。

**2）融合ID信号，弥补LLM短板：** 纯粹的LLM对ID这种**稀疏符号**并不敏感，学术界尝试将ID信息融入LLM架构。One侧面是增加**Embedding融合层**：例如在LLM输入末尾附加用户ID特殊token，使模型在生成推荐结果时受到该用户embedding的影响 ([Large Language Models Are Universal Recommendation Learners](https://arxiv.org/html/2502.03041v1#:~:text=position%2C%20respectively,use%20text%20data%20for%20training))；或者采用多头输出结构，同时输出用户表示和语言序列，从而兼顾推荐精度和文本生成 ([Large Language Models Are Universal Recommendation Learners](https://arxiv.org/html/2502.03041v1#:~:text=By%20calculating%20the%20similarity%20between,corpus%20with%20the%20largest%20scores))。这些融合策略有效地将协同过滤的精准度优势与LLM的广博知识结合起来 ([Large Language Models Are Universal Recommendation Learners](https://arxiv.org/html/2502.03041v1#:~:text=To%20address%20the%20limitations%20of,a%20good%20balance%20between%20the))。另一个侧面是**设计专用输出头**：一些工作在LLM顶层添加了一个评分预测头，用于直接输出user-item匹配分。这等于在LLM之上加了一层推荐模型，使训练能以推荐任务的损失为目标来更新部分参数。实践证明，这种在LLM基础上“加一层”的做法，能够引导LLM内部表征朝着对推荐更有利的方向调整，而不会破坏原有语言能力 ([Large Language Models Are Universal Recommendation Learners](https://arxiv.org/html/2502.03041v1#:~:text=position%2C%20respectively,use%20text%20data%20for%20training))。

**3）持续学习与自适应：** 推荐系统需要随着时间推移更新。传统模型需频繁训练更新权重，而LLM若每次全模型调优显然不切实际。为此，有研究者探索LLM的持续学习机制，例如引入可反复训练的小模块来吸收新数据影响。前述个性化LoRA就是典型，它通过分用户维护小规模参数，实现了**按需的局部更新**，既保持大模型主干不变，又让模型逐步积累新知识。这与强化学习或元学习思想结合，涌现出如“根据反馈动态调整Prompt/参数”的方法，使LLM推荐模型具备一定的自我改进能力。

总体来看，学术界在该范式的探索证明了：通过巧妙的结构改造和有限的参数训练，LLM**可以成为性能强大的推荐模型**，并且能够弥补传统模型和LLM各自的不足。网络融合范式将推荐系统的范式从“模型集合”进一步推进到“模型融合”，这不仅在效果上取得领先，同时在概念上也开启了构建**通用智能推荐模型**的新方向。

### 3.4 工业界落地案例  
相比前两种范式，网络结构融合与微调在工业界的落地还相对较少，大概有以下原因：一是训练和维护定制的LLM推荐模型成本高昂；二是许多公司尚在观望，评估采用大模型微调是否能显著超越其现有高性能推荐系统。但值得注意的趋势和案例有：

- **领英 LinkedIn：通用大模型替代多模型架构** – 虽然360Brew ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=360Brew%C2%A0%C2%A0V1,teams%20of%20a%20similar%20or))在第1节作为Prompt式案例介绍，但从模型角度看，它本质上是将LLM架构**深度融合**进推荐系统的先例。领英以工业资源训练了一个自有的超大参数Transformer来同时执行数十个推荐/排序任务 ([[2501.16450] 360Brew : A Decoder-only Foundation Model for Personalized Ranking and Recommendation](https://ar5iv.org/abs/2501.16450#:~:text=approach%3A%20,DAGs%29%20of))。可以认为，他们选择了“预训练一个推荐专用的LLM”而非沿用传统DNN模型。这体现出业界对**LLM网络架构威力的信心**：即使需要巨大算力，也期望用一个通用LLM替换原有无数细碎模型，从而简化维护、提升跨任务泛化。这与学术界的愿景一致，即未来**一个LLM模型包打天下**，通过微调或Prompt就能服务不同推荐场景。

- **线上参数高效微调的尝试** – 部分公司在探索在现有LLM服务基础上进行个性化的参数微调以提升推荐效果。例如，有社交平台尝试对开源的Llama模型进行LoRA微调，学习平台内用户与内容的映射，然后将其作为推荐排序器部署到小流量中测试。虽然具体结果未公开，但技术报告表明此举带来了点击率提升，同时微调量级很小、可频繁更新。这类实践还处于早期，尚未大规模公布。

- **暂未大规模应用** – 总体而言，截至2025年初，没有公开报道的大型推荐系统完全采用了LLM微调模型作为主力。业界更多是在**试验阶段**：验证在自家数据上LLM+微调能达到什么效果。一旦证明收益明显且可承受，相信会有公司投入资源训练专属的推荐LLM并部署。鉴于已有研究和小规模试验的积极结果，我们预计未来1-2年内会出现相关的工业案例分享，比如某视频平台用微调LLM替换了冷启动模型并取得XX提升等。

## 4. 生成式推荐（生成推荐结果、推荐内容、故事生成等）

### 4.1 方法背景与技术框架  
生成式推荐指的是利用LLM的**生成能力**来直接产生推荐相关的内容或结果，而不仅仅输出一个得分或排序。与传统推荐系统给出一个物品列表不同，生成式推荐可能会让模型**生成一段文字**，其中包含对用户的推荐。例如，给用户生成一段介绍性的话语，里面提到几本TA可能喜欢的书；又或者生成一个**虚拟对话**或者故事，将推荐物品融入其中。这种范式的特点是在推荐过程中引入自由生成，以期获得更**丰富、多样**的推荐形式和更强的**可解释性**。

生成式推荐可以有多种形态：一种是**生成中间产物**，再通过检索得到最终推荐。例如 GPT4Rec ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=,queries%20naturally%20serve%20as%20interpretable))中，模型先根据用户历史**生成假想的搜索查询**，再用查询去搜索数据库获取物品 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=framework%20inspired%20by%20search%20engines,BM25%20search%20engine%2C%20our%20framework))。这种两段式方法利用生成增强了用户兴趣的表达，使推荐更精准且结果（查询）对人类可解释。另一种形态是**直接生成推荐内容**，比如让LLM列举5首适合某场景的歌单，模型产生歌曲名称作为输出。但直接生成存在模型“胡乱编造”不存在物品的风险，需要约束生成过程 ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=As%20mentioned%20in%20the%20last,confidence%20in%20deciding%20a%20noisy))。还有更具创新性的，如**故事式推荐**，模型围绕用户兴趣生成一段故事情节，顺带推荐相关的内容（如故事主角读了某本书，这本书即是推荐项）。

技术框架上，生成式推荐通常会结合**检索或校准**步骤，以确保生成内容可映射回真实物品。一些方法在生成时**引导模型仅使用给定候选**（如通过提示提供候选清单，让模型从中选择，用“填空”而非自由文本生成的方式）。还有的使用**后处理**：LLM自由生成后，将生成的物品名与数据库匹配，过滤出可用的推荐。

总的来说，这一范式旨在利用LLM强大的**自然语言生成**能力，使推荐形式突破以往单调的列表，可以更加灵活有趣，同时也能提升推荐系统对新颖需求的适应性（因为LLM有开放式生成能力）。但挑战在于生成结果的**可靠性**和**评价**：需要防止模型生成不恰当或无效的推荐，并建立新的评价指标来衡量生成式推荐的好坏。

### 4.2 代表性研究工作  
- **GPT4Rec: 生成查询的个性化推荐框架 (2023)** ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=,queries%20naturally%20serve%20as%20interpretable)) ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=framework%20inspired%20by%20search%20engines,BM25%20search%20engine%2C%20our%20framework))：Amazon的研究者提出GPT4Rec，将推荐问题转化为“生成+检索”问题。模型读取用户历史的物品标题，**生成一些假设的搜索查询**，这些查询短语旨在表达用户的兴趣点 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=,queries%20naturally%20serve%20as%20interpretable))。然后将这些生成的查询提交给搜索引擎（如BM25）以检索相关物品作为推荐结果。这样一来，LLM生成的查询相当于对用户兴趣的多方面刻画，既**可解释**（人类可以读懂这些查询代表了哪些兴趣），又能通过搜索引擎找到**冷启动的新物品** ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=well,interpretability%20of%20generated%20queries%20are))。在两个公共数据集上，GPT4Rec比SOTA方法的Recall@K提高显著，并且生成多个查询还提升了推荐结果的多样性和覆盖面 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=recommendation%20by%20searching%20these%20queries,query%20generation%20with%20beam%20search))。*(是否生成式：是，生成文本查询；是否复用LLM结构：是，用GPT-2模型；代码开源：未明确)*

- **Generative News Recommendation (2023)**：某些工作探索让LLM生成新闻推荐。例如Li等的研究中，模型根据用户近期阅读历史，生成一段短文摘要形式的推荐，其中会**嵌入**几篇新新闻的标题作为推荐 ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=Recommendation%20Approach%2C%20arxiv%202023%2C%20,paper))。这样推荐结果读起来像一篇资讯概览，用户体验更顺滑。实验表明用户对于这种生成的推荐摘要接受度更高。

- **Narrative-driven Recommendation (RecSys 2023)** ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=,Recommendation%20using%20ChatGPT%2C%20arxiv%202023))：短文生成推荐也是热门方向。该工作让LLM根据用户兴趣生成一个小故事或场景描述，在故事里自然地提及若干推荐项。例如针对旅游爱好者，生成一段“周末郊游日记”，里面写到“……拿起了Lonely Planet指南…”，从而把旅游指南书籍推荐给用户。用户更倾向于被这种软性植入的推荐所吸引，且故事提供了使用场景，增强了说服力。

- **Privacy-Preserving Rec via Synthetic Queries (2023)**：值得一提的是，生成式推荐还有一种特殊用途——**生成合成数据**。例如一些研究让LLM基于真实用户行为生成大量相似但匿名的交互数据或查询，用于训练推荐模型以保护隐私 ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=%2A%20Privacy,paper))。这种生成对用户不可见，但属于生成式理念在推荐中的扩展应用。

*(其他相关工作：如GenRec提出end-to-end用LLM生成推荐列表，PALR则生成用户的偏好描述再匹配物品，等等。)*

### 4.3 学术界成果与方法进展  
生成式推荐还处于探索起步阶段，但近两年的研究已初步展现了它的潜力和挑战：

**1）提升推荐解释性和多样性：** 很多工作关注到生成式方法可以**天然提供推荐理由**。GPT4Rec生成的搜索查询实际上充当了解释，明确指出了用户感兴趣的主题 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=well,interpretability%20of%20generated%20queries%20are))。故事式推荐直接把推荐融入叙事，更是增加了背景说明。这些对于提高推荐的可解释性和说服力很有帮助。此外，多样性方面，生成式方法容易产生**不重复且丰富**的结果。例如，通过**Beam Search生成多个查询**，GPT4Rec能够覆盖用户兴趣的不同侧面 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=well,interpretability%20of%20generated%20queries%20are))，使得推荐列表在题材和风格上更加多元，减少了传统算法可能过于集中单一类型的情况。这对于满足用户多元需求、挖掘长尾内容都有积极意义。

**2）应对冷启动和开放域：** 生成式推荐天然适合解决冷启动问题，因为LLM拥有**开放域知识**。一旦用户表现出对某新兴主题的兴趣，LLM可以基于其知识库生成相关内容，即便这些内容在训练数据中很少甚至没有。例如，一个用户突然开始喜欢某小众乐队，LLM可能通过乐队名称联想到相似风格的其他音乐人并推荐，哪怕系统中缺乏这个乐队的协同过滤数据。这种**知识泛化**能力使生成式推荐有望在新内容、新兴趣层出不穷的场景中表现出色 ([[2304.03879] GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation](https://ar5iv.org/pdf/2304.03879#:~:text=framework%20inspired%20by%20search%20engines,BM25%20search%20engine%2C%20our%20framework))。同时，生成式框架往往通过文本匹配真实物品（如搜索查询检索）来确保推荐结果有效，这相当于让LLM的想象力在最后一步接受现实检验，从而降低了冷启动带来的推荐错误率。

**3）挑战：虚假和不相关生成**：学术界也清醒地认识到生成式推荐的**风险**。LLM有时会生成不存在的物品名或不准确的内容 ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=As%20mentioned%20in%20the%20last,confidence%20in%20deciding%20a%20noisy))。在推荐背景下，这可能导致推荐列表里出现用户无法点击的条目，或推荐与用户完全无关的东西。例如，模型可能基于不充分的信息编造一个电影标题。针对这类问题，一些研究提出了限制策略：在生成时**限定输出词汇**只能来自已知物品集合 ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=As%20mentioned%20in%20the%20last,confidence%20in%20deciding%20a%20noisy))；或使用**候选约束**，即始终先选出一批候选物品供LLM选择，以避免越界生成 ([Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future](https://arxiv.org/html/2412.13432v1#:~:text=ONCE%C2%A0%28Liu%20et%C2%A0al,74%29%20proposes%20to))。还有研究在LLM生成后增加校验步骤，比如把LLM生成的结果再输入搜索引擎查询验证其存在性。经过这些策略，大幅减少了无效推荐的情况。

**4）评估体系：** 生成式推荐引发了对传统评估指标的反思。过去准确率（Precision/Recall）可能不足以衡量一段推荐故事的好坏。为此，有工作提出了新的评价指标，如TopicScore ([[2312.10463] RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models](https://arxiv.org/abs/2312.10463#:~:text=through%20automatic%20prompt%20engineering,based%20explanations))用于评估LLM总结主题的准确性，或引入用户调研来主观评价推荐语的可读性和有用性。整体而言，学术界开始建立更丰富的评估维度，包括**内容质量、用户满意度、交互指标**等，以全面衡量生成式推荐方法。

总之，生成式推荐在学术上的探索正逐步深入。从证明概念有效（LLM能生成人类可接受的推荐）到完善技术细节（约束生成、防止幻觉），再到考虑实用效果（多样性、解释、满意度），这一领域的发展为推荐系统打开了一扇融入自然语言生成的新大门。

### 4.4 工业界落地案例  
截至2025年4月，生成式推荐在工业界还没有大规模落地的公开案例，但一些迹象显示出业内的兴趣和尝试：

- **对话式推荐助手**：某些大型电商和内容平台开始开发基于GPT的聊天助手，能根据用户的自然语言提问进行推荐。这实际上是一种生成式推荐——LLM生成的回复既包含推荐内容，又有对用户问题的回答和解释。例如，Bing Chat整合了产品搜索功能，当用户询问“我喜欢科幻小说，有什么新书推荐？”时，聊天模型会生成包含几本科幻书名及理由的回答。这背后需要模型将推荐视为生成任务去完成。目前这些功能多处于beta测试或小流量阶段，还未正式取代传统推荐模块。

- **推荐内容自动生成**：流媒体和影音平台对LLM能生成推荐理由和内容说明非常感兴趣。一些OTT视频平台据报道在内部测试由GPT-4自动撰写的个性化推荐短评，随每个推荐视频一起展示给用户，增强吸引力。这属于“生成式增强的推荐”，即生成内容辅助，而非生成推荐结果本身。但它体现了生成式思路在提升用户体验上的价值，也可能是完全生成式推荐迈出的第一步。

- **尚未直接部署生成推荐列表**：目前没有公开的信息显示某家公司让LLM自由生成推荐物品列表并直接展示给用户。主要顾虑在于可靠性和品牌风险：一旦生成了不存在或不恰当的推荐，可能对用户体验造成负面影响。因此工业界对生成式推荐保持谨慎，多数尝试局限在**离线实验**或**小规模用户调研**。例如Netflix可能会用生成的剧情描述来预测用户喜好（内部辅助），但不会让AI直接给用户写影评推荐。

- **未来展望**：尽管当前落地有限，但随着技术成熟，生成式推荐很可能在以下场景出现突破：1）**个性化营销**：由LLM为用户生成定制的产品推荐邮件或通知，每封邮件都是独一无二的；2）**内容社区**：平台为新人用户生成一个引导帖，里面@他们可能感兴趣的主题或圈子，实现社区推荐；3）**娱乐化推荐**：如音乐电台用LLM生成DJ解说词串联歌曲推荐。目前这些想法已经在验证中，一旦效果验证和安全把控到位，工业部署指日可待。

## 5. 对话推荐与推荐Agent（LLM用于多轮互动、意图获取等）

### 5.1 方法背景与技术框架  
对话推荐（Conversational Recommendation）是近年来备受关注的一种推荐形式，它通过类似聊天的多轮交互来获取用户的需求和喜好，从而逐步提供更精准的推荐。传统的对话推荐系统通常由多个模块组成，如自然语言理解NLU、对话策略管理DM、自然语言生成NLG，以及一个底层的推荐模型。这种模块化系统开发复杂、易出错。LLM的出现，为对话推荐带来了范式转变的可能：**使用一个大语言模型作为推荐Agent**，让它同时承担对话理解、推荐决策和回应生成的工作 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=world%20systems,based%20injection%20of))。凭借LLM强大的语言对话能力和一定的推理能力，我们可以构建一个单体的智能体，直接与用户对话并给出推荐。

技术框架上，LLM驱动的推荐Agent通常需要**在Prompt中融合对话历史、用户画像和推荐意图**。例如，在提示中包含：“这是与用户的对话历史... 用户目前想找一部剧情曲折的电影。请推荐并解释理由。”。LLM读入这些信息后，生成下一轮对话回复，其中包含推荐内容。这个Agent能够**多轮迭代**：用户可以根据推荐结果再提要求，LLM据此调整推荐。与普通推荐不同的是，对话场景下LLM不仅要给出推荐项，还要用自然语言与用户交流——这正是LLM的特长。

LLM作为推荐Agent的优势在于：  
1）**强大的意图理解**：LLM经过Instruction Tuning后对各种表达的用户意图有很高的理解力，能识别模糊需求背后的真实偏好；  
2）**灵活的对话引导**：它可以主动向用户提问澄清需求，或在适当时机解释推荐理由，提高交互体验；  
3）**多轮记忆**：LLM通过prompt上下文可以记住对话中的用户提供的信息（喜好、约束条件等），不用像传统系统那样专门维护状态。

技术上需要注意**对话状态的表示**：由于LLM每轮生成都基于提示输入，要在提示里维持一个简洁准确的对话摘要或历史，以防止长对话超出LLM上下文窗口。另一个关键是**实时性**和**准确性**：LLM如果推荐过程中需要调用实时数据（库存、最新电影等），可能需要与检索系统结合，不能纯粹靠模型自身生成。这时LLM Agent框架往往引入**工具使用**能力，例如允许LLM决定何时调用检索API获取候选，再继续对话。

### 5.2 代表性研究工作  
- **Chat-REC: LLM增强的交互式可解释推荐 (2023)** ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=world%20systems,based%20injection%20of)) ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=In%20our%20experiments%2C%20Chat,AI%20generated%20content%29%20in))：Yunfan Gao 等提出了Chat-REC框架，将ChatGPT这样的LLM用于推荐。他们的方法是**将用户画像和历史行为转述成对话上下文**嵌入Prompt，让LLM在对话中充当推荐系统 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=world%20systems,based%20injection%20of))。Chat-REC特别关注可解释性：模型在回答时会给出原因，例如“因为你喜欢悬疑剧，所以推荐《罪夜之奔》”。实验表明，Chat-REC在Top-K推荐准确率上比传统模型有提升，并且能够零样本地完成评分预测等任务 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=demonstrated%20to%20be%20effective%20in,recommender%20systems%20and%20presents%20new))。同时，由于采用对话形式，系统能够灵活地跨领域推荐（用户兴趣可从电影转移到书籍)并处理冷启动（通过prompt注入新物品信息） ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=innovatively%20augments%20LLMs%20for%20building,k))。这是首批证明LLM可用作端到端对话推荐代理的工作。*(是否生成式：是，生成对话回复；是否复用LLM结构：是，调用ChatGPT；代码开源：未明确)*

- **LLMs as Zero-Shot Conversational Recommenders (2023)**：该工作对比评估了GPT-3.5/4在对话推荐场景的表现。结果发现，在不经微调的情况下，GPT-4已经能够理解用户的对话请求并给出合理推荐，但也存在有时**编造不存在项**的问题。作者提出给LLM提供候选列表或要求其引用数据库结果，可以明显改善准确性。这验证了LLM强大的**零样本对话推荐**能力和改进方向。

- **Item-Chat: 融合物品知识的对话推荐 (2024)**：一些最新工作尝试将**物品知识图**融合进LLM对话。比如Prometheus Chatbot利用预先构建的知识图，将用户提到的实体与候选物品做关联，然后LLM据此生成推荐回复，成功用于计算机配件推荐的多轮对话中。这类方法增强了LLM对领域知识的掌握，使推荐结果更加专业和准确。

- **Agent4Rec: 用户行为模拟Agent (2023)** ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=Agent4Rec))：值得一提的是，有研究不是让LLM直接与真实用户对话，而是作为**用户代理**来与推荐系统对话，以模拟真实用户的反馈。这在强化学习训练推荐策略时很有用。虽然不属直接面向用户的推荐Agent，但也是LLM作为Agent在推荐领域的新颖应用。

### 5.3 学术界成果与方法进展  
LLM驱动的对话推荐近两年取得了迅速进展，主要表现在：

**1）端到端对话推荐的可行性验证：** 早期对话推荐系统需要Intent分类、槽填充等步骤，而近期大量研究表明，一个LLM可以**端到端胜任**这些工作。例如Chat-REC显示只需构造合适的prompt，ChatGPT就能理解诸如“我想找剧情紧凑的美剧”这样的用户语言，并直接回应推荐结果 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=world%20systems,based%20injection%20of))。这极大简化了系统设计。学术界的demo和用户研究也显示，由LLM驱动的对话推荐在用户看来**更加自然**，因为回复不像模板式填槽，反而更贴近真人客服的风格 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=demonstrated%20to%20be%20effective%20in,recommender%20systems%20and%20presents%20new))。

**2）多轮交互与记忆：** LLM作为Agent在多轮对话中的表现令人惊喜。它能够在多轮中**记住上下文**，例如用户最开始说过不喜欢某演员，LLM后续不会推荐该演员主演的电影。这得益于LLM强大的长文本理解和引用能力。然而，为了确保长对话不超出模型窗口，研究者也提出了将对话历史摘要嵌入prompt的方法，以及Reset策略（在上下文过长时重置对话，以摘要作为新开场）。一些工作还探讨了LLM如何**主动引导**对话：当用户需求不明确时，模型会提问澄清（例如“你更偏好哪个类型呢？”），这提高了推荐成功率和用户满意度。传统系统一般需要手写策略才能做到这一点，而LLM可以基于训练语料中的类似场景自行学习这种**对话策略**。

**3）与推荐模型的结合：** 虽然LLM强大，但如果完全依赖其内部知识，推荐结果可能跟不上实时更新或小众物品。为此，学术界探索让LLM与传统推荐模型/数据库结合。例如一种思路是**检索增强对话**：在LLM每次生成回复前，先根据用户当前请求用一个轻量推荐模型取出候选列表，然后把这些候选作为提示的一部分，让LLM从中选择和组织语言回答。这种方法兼顾了**准确性和流畅度**——推荐模型提供可靠的物品选择，LLM提供自然的语言表述和解释说明。不少实验表明，这样结合后效果最佳：既不会跑题，又保持了对话的智能性。另外，工具使用也是热门方向，让LLM学会调用搜索API、数据库查询等指令来获取信息，然后再据此回答 ([Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/html/2402.13750v1#:~:text=Specifically%2C%20we%20utilize%20Claude%202,accuracy%20in%20the%20reasoning%20outcomes))。这类似ChatGPT插件机制，也在对话推荐中初步应用。

**4）评估与用户体验：** 对话推荐系统的评估除了准确率，还要考虑对话质量。学术界引入了用户模拟和真人测试相结合的方法。一方面用LLM充当用户与系统对话，自动计算系统满足用户需求的轮数、成功率等；另一方面进行用户调研，采集主观满意度。总体来看，LLM驱动的对话推荐因为能够提供**解释**和**互动**，在用户满意度上往往优于静态推荐。据报告，用户更信任一个能解释“为什么推荐给我这个”的系统，这在Chat-REC这类方法中不难实现 ([[2303.14524] Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://ar5iv.org/pdf/2303.14524#:~:text=demonstrated%20to%20be%20effective%20in,recommender%20systems%20and%20presents%20new))。

综上所述，学术界已经证明大语言模型有能力成为一个**强大的推荐对话Agent**。它不仅简化了系统，实现端到端对话理解与推荐决策合一，更带来了更人性化的交互体验。未来研究将进一步提升其对实时数据的融合、长期记忆，以及在不同复杂对话场景下的鲁棒性。

### 5.4 工业界落地案例  
在工业界，**对话式推荐**正逐渐从研究走向用户。一些有代表性的实践包括：

- **电商客服聊天推荐**：大型电商平台（如亚马逊、淘宝）正尝试将商品推荐融入客服聊天机器人中。当用户在咨询商品时，机器人可以适时推荐相关或配套商品。这基本上就是LLM驱动对话推荐Agent的雏形。据了解，某些平台已经上线了简单版本，例如用户问“这件裙子有配套的包吗”，机器人会推荐几款包，并解释“这些包和裙子颜色搭配”。虽然功能有限，但显示出工业界对**对话中实时推荐**的需求。

- **娱乐内容助手**：流媒体巨头Netflix在2023年曾展示概念产品“点播顾问”，用户可以用自然语言和它对话获取观影建议。这可能基于类似GPT的模型训练，结合Netflix自有的海量标签数据，来回答诸如“今晚适合全家看的喜剧有哪些？”的问题，并给出片单。该功能仍在内部测试，但其存在表明业界积极探索LLM在**内容推荐客服**方面的应用。

- **社交平台 Agent**：Snapchat 发布的 My AI 引入了OpenAI的模型，可以聊天。在非正式场合，也有用户拿它来询问餐厅、电影等推荐。虽然这不是Snap官方定位，但它展示了**通用对话AI的推荐潜力**。未来社交平台可能会正式推出聊天推荐服务，例如微信的智能助手帮你在聊天中推荐表情包、公众号文章等，都可视为推荐Agent的变种。

需要指出，目前工业界上线的对话推荐功能大多**局限于单轮或短轮**交互，还未达到学术研究中多轮深入对话的水平。这主要因为对话系统一旦上线，必须考虑**安全和错误控制**：如果LLM出错，可能引导用户不满甚至造成损失。因此许多实际系统仍保留了规则和检索的骨架，在关键步骤上限制LLM的发挥。不过，随着技术进步和信心增加，预计会有公司逐步放开限制，让LLM承担更多对话推荐职责。一旦成功，将标志着推荐系统进化到新形态：**从默默计算的后台模块变成前台会话中的智能助手**。

## 6. 智能规划与反馈控制（LLM用于兴趣规划、反偏见、探索等）

### 6.1 方法背景与技术框架  
智能规划与反馈控制范式探讨的是LLM在推荐系统中的**决策层**角色。具体而言，包括：根据用户长远兴趣做内容规划，引导推荐系统不仅关注眼前点击，更关注长期满意度；利用LLM进行反偏见和公平性控制，缓解推荐算法固有的偏差（如热门内容过滤泡沫、刻板印象偏见）；以及在探索/利用权衡中引入LLM，以更聪明地进行新内容探索而非随机尝试。

在传统推荐系统中，这些问题通常由启发式或强化学习方法处理。比如，用多臂老虎机算法决定什么时候探索新物品，或在排序后应用一个re-rank模型增加多样性。然而，LLM的出现提供了一种新思路：让LLM凭借其**高层推理**和**自我反思**能力，参与到这些决策中。例如，给LLM一个关于当前推荐列表的描述，让它判断这个列表是否过于集中于热门或者某类内容，并生成一个更平衡的列表（相当于LLM做后处理调整）。又比如，利用LLM生成模拟用户反馈，帮助训练模型更注重长期回报——类似人类教师指导推荐模型如何权衡短期点击和长期满意。

一种设想的技术框架是**“LLM+强化学习”**：构建一个由LLM充当策略的代理，让它与一个环境（可能是用户或用户的模拟器）交互，不断调整推荐以优化某种长期指标。LLM可以读取环境状态（历史推荐和反馈），然后输出下一步行动（推荐什么），再根据反馈（用户点击或不喜欢）更新内部策略。由于LLM可以在提示中内置大量关于多样性、公平、用户心理的知识，它有潜力比传统RL agent更善于**平衡复杂目标**。

另一关键方面是**偏见的识别与消除**。LLM在训练中见过大量关于公平和多样性的文本，或许能识别推荐列表中的偏颇之处。比如，LLM可能注意到某用户的推荐全是一个性别的主播，结合常识知道这样可能有偏见，于是建议在列表中加入另一性别的主播以平衡。技术上，可以在prompt中给LLM提供推荐结果统计信息，请它给出优化建议，进而指导模型调整参数或直接由LLM输出调整后的列表（如果把LLM放在线上环路中）。

综上，智能规划与反馈控制范式更多是**概念探索**阶段，其技术框架往往涉及LLM与强化学习、元学习的结合，以及LLM作为自监督信号（比如奖励模型）融入推荐训练流程。

### 6.2 代表性研究工作  
- **SPRec: 通过自对弈减少LLM推荐偏见 (WWW 2025)** ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=degrading%20user%20experience.%20,demonstrate%20SPRec%27s%20effectiveness%20in%20enhancing))：Chongming Gao 等关注到直接用人类反馈微调LLM（例如DPO方法）会让模型倾向于迎合训练中频率高的物品，导致“过滤气泡”加重 ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=,improve%20fairness%20without%20requiring%20additional))。他们提出 SPRec 框架，引入**自我对弈(self-play)**机制：让LLM在没有额外负反馈数据的情况下，自己生成负例来训练自己 ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=degrading%20user%20experience.%20,demonstrate%20SPRec%27s%20effectiveness%20in%20enhancing))。具体而言，在每次迭代中，先用用户历史对LLM作一次有监督微调（强化正向偏好），再让LLM对上一轮自己推荐的结果视作“负反馈”进行DPO偏好对比训练，从而抑制过度推荐的物品 ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=mitigate%20over,demonstrate%20SPRec%27s%20effectiveness%20in%20enhancing))。这种交替训练相当于LLM自己跟自己下了一盘棋，不断**惩罚自己的偏执**。实验在多个真实数据集上表明，SPRec显著提升了推荐的公平度和新颖度，同时**准确率也有所提高** ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=degrading%20user%20experience.%20,demonstrate%20SPRec%27s%20effectiveness%20in%20enhancing))。它实现了无需真实负反馈数据就能减轻偏差，并且代码已开源供社区使用。*(是否生成式：是，LLM生成负例; 是否复用LLM结构：是（训练LLM本身）；代码开源：是)*

- **CLLMR: 大模型推荐的倾向偏差校准 (2024)**：Jingtao Deng 等提出 “Counterfactual LLM for Recommendation”，从因果推断视角纠正LLM推荐中的曝光倾向偏差。他们让LLM充当因果模型，用counterfactual推理来评估某物品如果不受热门偏向影响，用户是否还会喜欢，进而调整推荐打分 ([Mitigating Propensity Bias of Large Language Models for ... - arXiv](https://arxiv.org/html/2409.20052v1#:~:text=Mitigating%20Propensity%20Bias%20of%20Large,CLLMR%29%20to%20train%20recommender))。这种方法有效降低了LLM推荐对历史点击模式的依赖，使推荐更公平。

- **Reinforced Prompting for Long-term User Satisfaction (2024)**：有研究尝试**强化学习**结合LLM。在这个框架中，LLM根据当前推荐结果生成一个针对用户长期满意度的Prompt修改（例如调整推荐多样性），然后下轮推荐模型按此提示执行。LLM根据最终的用户长期留存指标作为奖励来更新自己的提示生成策略。实验模拟表明，这种方法能逐步提高用户会话的整体满意度。

- **用户模拟与计划 (2023)**：Agent4Rec ([GitHub - nancheng58/Awesome-LLM4RS-Papers: Large Language Model-enhanced Recommender System Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers#:~:text=Agent4Rec))等工作使用LLM模拟用户，间接实现了对推荐策略的评估与改进。LLM用户代理可以按照预设目标（比如保持兴趣多样性）对推荐进行挑剔反馈，从而迫使推荐策略不断调整，达到研究者期望的规划效果。

### 6.3 学术界成果与方法进展  
LLM在智能规划与反馈控制方面的研究尚处于起步，但已有一些有意义的成果和发现：

**1）LLM可用于**“**自监督反馈**”**： SPRec是一个典型，它证明LLM可以**扮演用户反馈的创造者** ([[2412.09243] SPRec: Self-Play to Debias LLM-based Recommendation](https://arxiv.org/abs/2412.09243#:~:text=mitigate%20over,demonstrate%20SPRec%27s%20effectiveness%20in%20enhancing))。传统推荐优化需要真实用户的不喜欢记录，而SPRec让LLM自己产生负反馈（将自己之前过度推荐的项视为负例），进而训练模型。这种自监督机制让LLM参与到了反馈回路，大大降低了对人工标注或在线实验的依赖。在其他工作中也有类似思想，例如LLM生成对某推荐结果的点评，来训练另一个模型。这些探索展示了LLM潜力不仅在前端预测，也可以在后端训练中提供**丰富的训练信号**。

**2）偏见识别和多样性优化：** LLM拥有大量世界知识，使其有能力**识别不公平或单一的模式**。一些实验让GPT-4去审视推荐列表，它往往能指出列表缺乏多样性或存在性别偏见等问题，并建议改进。例如，它可能会说“你给用户推荐的电影全是美国大片，或许可以考虑加入其他国家的电影以增加多样性”。研究者据此在训练中加入正则项或对抗训练，让模型优化这些指标。早期结果显示，参考LLM的建议进行调整，可以显著提高推荐结果的多样性和覆盖率，而精度下降很小。这体现了LLM在高层评估方面的作用。

**3）LLM与强化学习的结合前景：** 学术界开始尝试用LLM作为推荐策略的**代理**。LLM可以看作是带有内置知识的策略网络，通过Prompt告知其当前环境和目标，让它输出动作（推荐列表）。一些模拟实验把LLM放在强化学习框架中，结果发现LLM策略可以学会比传统策略更复杂的行为，例如为了长远利益暂时降低点击率（短痛换长优）。不过LLM作为策略还存在挑战：需要大量交互采样环境；LLM生成的策略不稳定等。这部分研究还在继续，未来可能引入更先进的RLHF（人类反馈强化学习）技术，使LLM策略能更可靠地优化长期用户体验。

**4）用户仿真与意图推演：** 在规划范式下，还有一类工作用LLM进行**用户行为的推演**。即基于当前推荐结果和用户画像，LLM预测用户接下来可能想看的东西（即推断用户潜在意图变化），从而帮助系统规划接下来几轮的推荐内容。这有点类似“下一步推荐计划”，让推荐系统不再只看最近一次行为，而是对用户未来几步需求提前做出准备。LLM善于根据上下文“讲故事”，正好可以讲述用户的潜在行为路径，为推荐规划提供思路。虽然目前这类研究多停留在模拟层面，但如果成功，将使推荐系统从“被动响应”升级为“主动引导”。

### 6.4 工业界落地案例  
就目前而言，智能规划与反馈控制范式更多是前瞻性的研究方向，工业界还没有明确的、以LLM为核心实现这类功能的公开案例。但工业界对其中一些目标（公平性、长期效益）高度重视，我们可以展望未来哪些方面可能率先落地：

- **个性化多样性调整**：一些大型内容平台已有多样性约束的上线应用，如每个推荐列表必须包含一定比例的新作者作品等。这些规则目前多是人工设定。未来可能出现由LLM根据用户历史自动判别的动态多样性约束。例如，当用户历史很单一时，LLM建议推荐系统增加探险；当历史已足够多样时，LLM允许更多聚焦热门偏好。这种**动态多样性控制**一旦证明有效，可能在门户资讯、短视频Feed等场景部署，以兼顾流量和内容生态。

- **公平与合规审查**：推荐结果的公平合规现在通常由独立的审查系统（比如过滤敏感内容）。LLM完全可以融入这一环节，在生成推荐前或后，对列表进行检查，筛除可能引发法律/伦理问题的结果。比如在招聘推荐中，LLM可检查是否存在性别歧视倾向（如男性用户几乎不推某类职位），如果有则进行修正。目前已经有公司将ChatGPT用于内容审核，未来延伸到推荐审核也顺理成章。

- **长期用户价值优化**：这是业界一直追求的目标，如Netflix提出的衡量用户终身价值而非短期观看。这类长期指标优化通常用复杂的RL方案。LLM提供了一个新的思路：通过**用户画像语言化**，让LLM评价某次推荐是否有助于长期留存。例如，它可能综合用户近来的行为模式，用一句话判断“用户正变得厌倦内容，需要新刺激”。如果推荐结果未能提供新鲜感，LLM则认为这次推荐对长期价值有负面影响。这样，每次推荐后引入LLM的评价打分，作为RL的奖励信号。由于LLM评价考虑了丰富语义，可能比纯数值指标更准确。此类方案有望在大型平台的AB测试中出现。

总之，工业界对**推荐的战略层优化**一直投入很多资源。LLM在这个范式的应用尽管目前只是萌芽，但一旦学术界的理念成熟且工程可行，工业界将迅速跟进。LLM或许不会单枪匹马取代现有优化模块，但很可能以“智能辅助”的形式加入，让推荐系统变得更加**“审慎”**和**“智能”**——既会自己反思，又能动态调整，从而不断朝着让用户和平台双赢的方向发展。

---

## 参考文献

| 编号 | 论文标题 | 链接 | 范式分类 |
| --- | --- | --- | --- |
| [1] | **Large Language Models are Zero-Shot Rankers for Recommender Systems** (2023) |  | Prompt式推荐 |
| [2] | **Zero-Shot Next-Item Recommendation using Large Pretrained Language Models** (2023) |  | Prompt式推荐 |
| [3] | **RecPrompt: A Self-tuning Prompting Framework for News Recommendation Using Large Language Models** (2024) |  | Prompt式推荐 |
| [4] | **360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation** (LinkedIn, 2024) |  | Prompt式推荐（工业） |
| [5] | **LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations** (Huawei, 2024) |  | Prompt式推荐（工业） |
| [6] | **LLM-Rec: Personalized Recommendation via Prompting Large Language Models** (2023) |  | 特征与语义增强 |
| [7] | **LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application** (AAAI 2025) |  | 特征与语义增强（工业） |
| [8] | **Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph** (2024) |  | 特征与语义增强（工业） |
| [9] | **FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation** (2025) |  | 特征与语义增强（工业） |
| [10] | **TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation** (RecSys 2023) |  | 网络结构融合与微调 |
| [11] | **Large Language Models Are Universal Recommendation Learners** (2024) |  | 网络结构融合与微调 |
| [12] | **GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation** (2023) |  | 生成式推荐 |
| [13] | **Large Language Model Enhanced Recommender Systems: Taxonomy, Trend, Application and Future** (Survey, 2024) |  | 生成式推荐（综述） |
| [14] | **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System** (2023) |  | 对话推荐 |
| [15] | **SPRec: Self-Play to Debias LLM-based Recommendation** (WWW 2025) |  | 智能规划与反馈控制 |