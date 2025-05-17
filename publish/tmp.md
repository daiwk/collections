[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

[https://github.com/deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)

FFN部分的MoE，相比传统的[GShard](https://arxiv.org/pdf/2006.16668.pdf) 等MoE架构，DeepSeekMoE用了**更细粒度的专家分配机制**，并将部分专家设置为**共享专家**。

$\mathbf{u}_t$表示FFN输入的第$t$个token，$N_s$是shared experts的数量，$N_r$是routed experts的数量，$K_r$表示被激活的router专家数量 

$$
\begin{aligned}
& \mathbf{h}_t^{\prime}=\mathbf{u}_t+\sum_{i=1}^{N_s} \operatorname{FFN}_i^{(s)}\left(\mathbf{u}_t\right)+\sum_{i=1}^{N_r} g_{i, t} \operatorname{FFN}_i^{(r)}\left(\mathbf{u}_t\right), \\
& g_{i, t}= \begin{cases}s_{i, t}, & s_{i, t} \in \operatorname{Topk}\left(\left\{s_{j, t} \mid 1 \leqslant j \leqslant N_r\right\}, K_r\right), \\
0, & \text { otherwise },\end{cases} \\
& s_{i, t}=\operatorname{Softmax}_i\left(\mathbf{u}_t^T \mathbf{e}_i\right),
\end{aligned}
$$

device-limited routing：保证每个token的目标专家**分布在最多$M$个设备上**

+ 首先选择有最高相关性（即上面的$s_{i,t}$）的$M$个设备出来
+ 再从这$M$个设备里选出top-K个专家出来

负载均衡的辅助loss：

+ expert-level的负载均衡loss：$\alpha_1$是超参

$$
\begin{aligned}
\mathcal{L}_{\text {ExpBal }} & =\alpha_1 \sum_{i=1}^{N_r} f_i P_i, \\
f_i & =\frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}(\text { Token } t \text { selects Expert } i), \\
P_i & =\frac{1}{T} \sum_{t=1}^T s_{i, t},
\end{aligned}
$$

+ device-level的负载均衡loss：把所有routed专家分成$D$组$\left\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_D\right\}$，并把每一组放到一个单独的设备上：

$$
\begin{aligned}
\mathcal{L}_{\text {DevBal }} & =\alpha_2 \sum_{i=1}^D f_i^{\prime} P_i^{\prime} \\
f_i^{\prime} & =\frac{1}{\left|\mathcal{E}_i\right|} \sum_{j \in \mathcal{E}_i} f_j \\
P_i^{\prime} & =\sum_{j \in \mathcal{E}_i} P_j
\end{aligned}
$$

+ 通信负载均衡loss：如果一个设备收到的token比较多，它的通信代价也比较大。因为要选$M$个设备出来，总共有$T$个词，所以，鼓励每个设备最多传输$MT$个隐层状态去其他设备，同时从其他设备接收到$MT$左右个的隐层状态。

$$
\begin{aligned}
\mathcal{L}_{\text {CommBal }} & =\alpha_3 \sum_{i=1}^D f_i^{\prime \prime} P_i^{\prime \prime} \\
f_i^{\prime \prime} & =\frac{D}{M T} \sum_{t=1}^T \mathbb{1}(\text { Token } t \text { is sent to Device } i) \\
P_i^{\prime \prime} & =\sum_{j \in \mathcal{E}_i} P_j
\end{aligned}
$$
