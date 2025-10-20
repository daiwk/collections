+ DeepEncoder：提取图像特征，并将视觉表示进行Token化和压缩，共380M参数
    + 80M的SAM-base：原始图像切成n个patch，单个patch大小为16x16，然后输入给SAM进行local attn
    + 2层的卷积模块：对视觉Token进行16倍的下采样，即得到n/16个token。参考[Vary: Scaling up the vision vocabulary for large vision-language model](https://arxiv.org/pdf/2312.06109)，每个卷积层的核大小为3，步长为2，pad为1，通道数从256增加到1024。具体过程如下：
        + 假设输入一张1024x1024的图像
        + 输入给sam的是$1024\times 1024/16 \times 16 \times 16=64\times 64 \times (16\times 16)$，即$64\times 64$个patch
        + 从sam输出的应该是$64\times 64 \times 256$。
        + 经过第一个conv后，宽度变成$(64-2\times 1 - 3)/2 +1=32$，所以shape是(32,32,256)
        + 再经过第二个conv后，宽度变成$(32-2\times 1-3)/2+1=16$，所以shape是(16,16,1024)，即token数变成了$16\times 16=256$，相比原来的$64\times 64$而言就是缩小了16倍。
    + 300M的CLIP-large：进行global attn
+ DeepSeek3B-MoE-A570M解码器：根据图像Token和prompt生成所需的结果，用来重建文本表示。570M in 3B的MoE，64个router专家激活6个，还有2个共享专家
