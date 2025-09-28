[RecoWorld: Building Simulated Environments for Agentic Recommender Systems](https://arxiv.org/pdf/2509.10397)

图1

用户模拟器输入用户历史行为，产出一个instruction，丢给agentic recsys，产出推荐list，用户模拟器产出用户action，同时再产出新的instruction，再丢给agentic sys，再产出新的推荐list，直到整个trajectory结束，产出整个轨迹的奖励，拿来用rl训练agentic recsys。

图2和图3

用户模拟器的建模，可以是文本llm，还可以是多模态llm、把item换成sid的llm（需要先重新tune出一个llm才能用）。还加了memory模块（记录长期兴趣）、reasoning、midset update（如第二张图，给用户推了a，用户会是一个什么心态，下一步希望什么）

图4

推荐系统是一个自动化的agent，包括4个能力：perception、reasning&planning、action(工具使用)、memory，其参数可以用RL来训练，环境的状态就是上面的用户的mindset，action就是推荐的item list，最终奖励就是整个多轮交互的总reward

