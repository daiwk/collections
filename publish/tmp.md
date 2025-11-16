
定义：

+ 输入$n$个token $x_0, x_1, \ldots, x_{n-1}$，每个token包括item和对应的action，即$\left(\Phi_0, a_0, \ldots, \Phi_{n_c-1}, a_{n_c-1}\right)$，也就是$n=2 n_c$
+ 精排有$m$个候选 $\Phi_0^{\prime}, \ldots, \Phi_{m-1}^{\prime}$，Micro batchsize $b_m$，可以分成$\text { numMicrobatches }=\left(m+b_m-1\right) / / b_m$份
+ $\text { attnMask }=L_{n+b_m}$，直接加到attention logit上，是一个下三角矩阵，即下三角是0，其他的是$-\infty$；此外，$\operatorname{attnMask}[i, j]=-\infty \text { for } i, j \geq n, i \neq j$，表示大于n的那些item之间是互相看不到的，如下（白色可见，深色不可见）：

进行一次计算：$\left(a_0^{\prime}, a_1^{\prime}, \ldots, a_{b_m-1}^{\prime}\right), k v C a c h e \leftarrow f\left(e m b L a y e r\left(\left(x_0, x_1, \ldots, x_{n-1}, \Phi_0^{\prime}, \ldots, \Phi_{b_m-1}^{\prime}\right)\right), \varnothing, \text { attnMask }\right)$，其中$\text { predictions }=\left(a_0^{\prime}, a_1^{\prime}, \ldots, a_{b_m-1}^{\prime}\right)$

算完第一个microbatch后，得到这n个token的kvcache。算下一个microbatch的时候，把这个的microbatch拼到这n个token后面，复用kvcache和attnmask，得到这个microbatch的预估值，以此类推。
