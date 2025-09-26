+ $\mathbf{X} \in \mathbb{R}^{T \times d_{\text {model }}}$
+ $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{T \times h \times d_h}$，共$h$个head，每个head的dim是$d_h=d_k=d_{model}/h$，$\boldsymbol{W}^O \in \mathbb{R}^{\left(h \cdot d_h\right) \times d_{\text {model }}}$
+ $\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{T \times d_h}$表示第$i$个head
+ $\boldsymbol{W}_i^Q, \boldsymbol{W}_i^K, \boldsymbol{W}_i^V \in \mathbb{R}^{d_{\text {model }} \times d_k}$表示第$i$个head的权重
+ $\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{h \times d_h}$表示第$t$个token的Q/K/V
+ 向量外积：给定$\mathbf{a} \in \mathbb{R}^m, \mathbf{b} \in \mathbb{R}^n$，外积$\mathbf{a} \otimes \mathbf{b}=\mathbf{C} \in \mathbb{R}^{m \times n}$，其中$C_{i j}=a_i b_j$
+ 矩阵外积：矩阵$\mathbf{C} \in \mathbb{R}^{m \times n}$，定义$\operatorname{vec}(\mathbf{C}) \in \mathbb{R}^{m n}$，即把矩阵flatten成一个长的**列向量**；也可以这么理解，$\mathbf{C}=\left[\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_n\right]$，有n个列向量，那么，$\operatorname{vec}(\mathbf{C})=\left[\mathbf{c}_1^{\top}, \mathbf{c}_2^{\top}, \ldots, \mathbf{c}_n^{\top}\right]^{\top}$

TPA本质就是把$\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t \in \mathbb{R}^{h \times d_h}$进行分解，即$h\times d_h$的矩阵拆成$h$的a向量，和$d_h$的b向量的外积，还搞了R组的(a,b)，算R次外积并取平均值。

$$
\begin{gathered}
\mathbf{Q}_t=\frac{1}{R_Q} \sum_{r=1}^{R_Q} \mathbf{a}_r^Q\left(\mathbf{x}_t\right) \otimes \mathbf{b}_r^Q\left(\mathbf{x}_t\right), \quad \mathbf{K}_t=\frac{1}{R_K} \sum_{r=1}^{R_K} \mathbf{a}_r^K\left(\mathbf{x}_t\right) \otimes \mathbf{b}_r^K\left(\mathbf{x}_t\right), \\
\mathbf{V}_t=\frac{1}{R_V} \sum_{r=1}^{R_V} \mathbf{a}_r^V\left(\mathbf{x}_t\right) \otimes \mathbf{b}_r^V\left(\mathbf{x}_t\right),
\end{gathered}
$$


其中，$\mathbf{a}_r^Q\left(\mathbf{x}_t\right), \mathbf{a}_r^K\left(\mathbf{x}_t\right), \mathbf{a}_r^V\left(\mathbf{x}_t\right) \in \mathbb{R}^h, \mathbf{b}_r^Q\left(\mathbf{x}_t\right), \mathbf{b}_r^K\left(\mathbf{x}_t\right), \mathbf{b}_r^V\left(\mathbf{x}_t\right) \in \mathbb{R}^{d_h}$