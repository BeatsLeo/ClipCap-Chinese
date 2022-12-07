## 基于ClipCap实现中文Image Caption

一个项目用于两个课程设计也挺不错`^_^`



### 主体思路

通过图像特征提取模型将图片转化为向量，再通过映射网络将所提取的向量转化为文本生成前缀，将文本生成前缀prefix_embeds与constant_embeds进行拼接作为输入传进GPT2模型，从而生成文本。



### 图像特征提取

利用ClipCap在极大的数据集进行预训练所得到的图像特征提取模型



### 文本生成

利用GPT2在极大的数据集进行预训练所得到的文本生成模型



### 图像特征到文本特征映射网络

* MLP
* MLP + GPT2微调
* Transformer
* Transformer + GPT2微调



### 参考文献

[Mokady, Ron, Amir Hertz and Amit H. Bermano. “ClipCap: CLIP Prefix for Image Captioning.”ArXivabs/2111.09734 (2021): n. pag.](https://arxiv.org/abs/2111.09734)

