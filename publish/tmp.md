[ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users](https://arxiv.org/pdf/2510.09393)

+ stage1：Hierarchical Semantic User Group Generation
    + Semantic Profile Synthesis via LLM：输入用户静态信息、各种时间窗口的行为类目序列、搜索序列，输出core identity、interest points和consumption philosophy
    + Hierarchical Group Construction：
        + 通过Qwen3-embedding-8B等模型将profile编码成emb，
        + 通过RQ–KMeans搞出M级semantic group id
+ stage2：构建Group-aware Hierarchical Representation
    + Hierarchical Group ID Fusion：M级group id对应的emb按顺序融合，再concat过mlp
    + Group Attribute Completion：当前group里用户的静态特征+统计特征聚合起来，相当于对低活用户可以有一个group级的初始化；可以预先算好
    + Group Behavioral Sequence Construction：
        + Group Interest Identification：group内所有用户的购买历史里，聚合出top-k的类目
        + Group Sequence Construction：top-k类目中，保留每个类目里的头部item，同时用group内的平均购买时间来排序
+ stage3：Group-aware Multi-granularity(多粒度) Module
    + individual channel：作为teacher，U和I过tower
    + group channel：作为student，G和I过mlp过tower
    + group channel的输入通过fusion tower输入给individual channel，但对group channel有stop grad
    + 用户活跃度特征过nn得到一个emb，然后输出2个gate：
        + merge用：两个channel的logit进行merge的时候用上面的gate加权，$z_{\text {fused }}=\left(1-\alpha_{\text {fusion }}\right) \cdot z_{\text {ind }}+\alpha_{\text {fusion }} \cdot z_{\text {group }}$
        + 蒸馏用：mse加上一个margin，另外有一个置信度的gate（用户购买数大于某阈值&teacher输出过完sigmloid比0.5大一定阈值）$g_{\text {qual }}=\mathbb{I}\left(\left|S_u^{\text {buy }}\right| \geq \theta_{\text {act }} \wedge\left|\sigma\left(z_{\text {ind }}\right)-0.5\right|>\theta_{\text {conf }}\right)$，还有上面的gate，$\mathcal{L}_{\mathrm{KD}}=g_{\text {qual }} \cdot \alpha_{\text {distill }} \cdot \mathcal{L}_{\text {margin }}$

