优化MFU的主要手段：

1. 让模型更加计算bound
    + batchsize变大：可以减少kernel启动次数
    + kernel融合：把多个细碎算子合并成一个大算子
    + 增加并行度：让TP变大（例如从1变成8），这样模型能算更大矩阵乘，而大GEMM的MFU是高的
    + 减少CPU/GPU之间等待：例如dataloader读数据慢
2. 加快通信：
    + 提高通信带宽：卡内nvlink通信、卡间网络通信
    + 减小通信量：例如通信量```ZeRO1<ZeRO2<ZeRO3```
    + 更好的通信调度：把计算和通信overlap，即在GPU上跑某个compute kernel（例如GEMM）时，不阻塞通信（NCCL allreduce/allgather）等，例如Megatron-LM的```--overlap-grad-reduce```和```--overlap-param-gather```
