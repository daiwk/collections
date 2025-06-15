## GAVE

[sigir'25「快手」生成式出价-冠军方案｜Generative Auto-Bidding with Value-Guided](https://mp.weixin.qq.com/s/fttTPY6Q30gWaSwcoIpUsA)

[Generative Auto-Bidding with Value-Guided Explorations](https://arxiv.org/pdf/2504.14587)

用序列（Decision Transformer）代替RL，再利用离线强化的方法去弥补没有模拟环境的缺点。DT的一些其他应用：

+ RAG：[Retrieval-Augmented Decision Transformer: External Memory for In-context RL](https://arxiv.org/pdf/2410.07071)
+ 序列推荐：[Sequential Recommend for Optimizing Both ImmediateFeedback and Long-term Retention](https://arxiv.org/pdf/2404.03637)
+ 重排：[PISDR: Page and Item Sequential Decision for Re-ranking Based on Offline Reinforcement Learning](https://raw.githubusercontent.com/mlresearch/v260/main/assets/yuan25a/yuan25a.pdf)

MDP假设是状态独立的，本质上忽略了竞价序列中的**时序依赖**关系。

自动出价：一般广告主会设置一个最低ROI（即价值/成本）作为约束，或者说是最大平均成本（CPA，即成本/价值），广告主给定的约束假设是$C$。

## DT

DT的输入(reward, state, action)三元组的序列，输出是预测的action，

**注意：对于t时刻来讲，输入的是0到t-1时刻的r+s+a，但只输入了t时刻的r和s**

+ $s_t$：历史出价策略、剩余预算、广告的在线时间等
+ $a_t$：出价因子，在广告的value上乘以a，$b_t=a_tv_t$
+ $rw_t$：**t到t+1**所有候选impression的总value，$r w_t=\sum_{n=0}^{N_t} x_{n_t} v_{n_t}$
+ RTG（return to go） $r_t$：现在**到未来**的总收益，$r_t=\sum_{t^{\prime}=t}^T r w_{t^{\prime}}$

## GAVE

![](../assets/gave.png)

预测如下4个值：

+ 当前时刻的action：$\hat{a}_t$
+ 当前时刻的探索系数：$\hat{\beta}_{t}$，其实是就是在$a_t$前面乘一个$\beta$得到$\tilde{a}_t=\hat{\beta}_t a_t$，然后约束一下$\beta$的范围在0.5到1.5之间（sigmoid(x)+0.5就行）可以减轻OOD(Out-of-Distribution)问题。

假设$\tilde{r}_{t+1}$是$\tilde{a}_{t+1}$的RTG，而$\hat{r}_{t+1}$是$a_t$的RTG，定义了如下的w（即$\tilde{r}_{t+1}$比$\hat{r}_{t+1}$大的概率）

$$
\left\{\begin{array}{l}
\left.\left.\tilde{r}_{t+1}=\operatorname{GAVE}\left(r_{t-M}, s_{t-M}, a_{t-M}, \ldots, r_t, s_t, \tilde{a}_t\right)\right)\right) \\
w_t=\operatorname{Sigmoid}\left(\alpha_r \cdot\left(\tilde{r}_{t+1}-\hat{r}_{t+1}\right)\right)
\end{array}\right.
$$

此外，还加了如下的辅助损失，其中$w'$和$\tilde{a}'$表示的是不更新梯度的$w$和$\tilde{a}$。第一个$L_r$让$\hat{r}_{t+1}$接近真实值，第二个$L_a$表示如果$w_t>0.5$，即$\tilde{r}_{t+1}$比$\hat{r}_{t+1}$大得比较多，第二项占主导，即让预测的action去你和探索的action $\tilde{a}_t'$，反之让预测的action去你和实际的action $a_t$

$$
\left\{\begin{array}{l}
L_r=\frac{1}{M+1} \sum_{t-M}^t\left(\hat{r}_{t+1}-r_{t+1}\right)^2 \\
L_a=\frac{1}{M+1} \sum_{t-M}^t\left(\left(1-w_t^{\prime}\right) \cdot\left(\hat{a}_t-a_t\right)^2+w_t^{\prime} \cdot\left(\hat{a}_t-\tilde{a}_t^{\prime}\right)^2\right)
\end{array}\right.
$$

+ 下一时刻的RTG值：$\hat{r}_{t+1}$，其实就是把CPA的约束加到RTG的计算里来：

$$
\left\{\begin{array}{l}
C P A_t=\frac{\sum_i^{I_t} x_i c_i}{\sum_i^{I_t} x_i v_i} \\
\mathbb{P}\left(C P A_t ; C\right)=\min \left\{\left(\frac{C}{C P A_t}\right)^\gamma, 1\right\} \\
S_t=\mathbb{P}\left(C P A_t ; C\right) \cdot \sum_i^{I_t} x_i v_i \\
r_t=S_T-S_{t-1}
\end{array}\right.
$$

+ 下一时刻的价值分：$\hat{V}_{t+1}$

目标是如何高效探索，用了一个expectile regression(IQL里的思想，[Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169))，如下，其中$L_2^\tau(y-m(x))$是一个loss函数，用模型$m(x)$来预测$y$的分位数$\tau \in (0,1)$，$\tau=0.85$就是让预估价值$\hat{V}_{t+1}$去预测top百分之85的reward $r_{t+1}$

$$
\begin{aligned}
L_e & =\frac{1}{M+1} \sum_{t-M}^t\left(L_2^\tau\left(r_{t+1}-\hat{V}_{t+1}\right)\right) \\
& =\frac{1}{M+1} \sum_{t-M}^t\left(\left|\tau-\mathbb{1}\left(\left(r_{t+1}-\hat{V}_{t+1}\right)<0\right)\right|\left(r_{t+1}-\hat{V}_{t+1}\right)^2\right)
\end{aligned}
$$

而对于探索的reward $\tilde{r}_{t+1}$来讲，则不更新价值的梯度，即$\hat{V}_{t+1}^{\prime}$，只需要约束reward在价值附近就行

$$
L_v=\frac{1}{M+1} \sum_{t-M}^t\left(\tilde{r}_{t+1}-\hat{V}_{t+1}^{\prime}\right)^2
$$

