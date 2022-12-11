# 基于ClipCap实现中文Image Caption

一个项目用于两个课程设计也挺不错`^_^`



## 项目简介

本项目从学校课程设计中诞生，参考[ClipCap](https://arxiv.org/abs/2111.09734)进行中文图像标注工作。

该项目所使用数据集为[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)，由于该数据集里面全是人物相关图像，所以导致本项目训练所得模型对包含人物的图像标注效果较好，而不含人物的图像效果较差。



### 效果示例

<img src="E:\QQDownload\DIP_NLP\code\ClipCap\test_images\football.jpg" alt="football" style="zoom: 50%;float:left;" />

“两个人的旁边有一个双臂张开的男人跑在足球场上。”
“两个穿着运动装的男人在运动场上庆祝。”
“三个人的旁边有一个双手握拳的男人奔跑在球场上。”
“两个人的旁边有一个双手握拳的运动员在球场上奔跑。”
“两个人旁有一个抬着右手的男人走在绿茵茵的球场上。”



### 使用方法

进入`ClipCap`文件夹，在当前路径下，运行`use.py`，指令如下：

`python use.py -m ./models/clipcap_mlp_finetune.model -i ./test_images/football.jpg`

`-m`：训练好的模型存放路径。

`-i`：需要标注的图片路径。



### 环境依赖

见`requirements.txt`。



### 主体思路

通过图像特征提取模型将图片转化为向量，再通过映射网络将所提取的向量转化为文本生成前缀，将文本生成前缀prefix_embeds与constant_embeds进行拼接作为输入传进GPT2模型，从而生成文本。



#### 图像特征提取

利用Clip在极大的数据集进行预训练所得到的图像特征提取模型的权重(ViT-L/14)



#### 文本生成

利用GPT2在极大的数据集进行预训练所得到的文本生成模型的权重



#### 图像特征到文本特征映射网络

* MLP
* MLP + GPT2微调（本项目采用）
* Transformer
* Transformer + GPT2微调



### 参考文献

[Mokady, Ron, Amir Hertz and Amit H. Bermano. “ClipCap: CLIP Prefix for Image Captioning.”ArXivabs/2111.09734 (2021): n. pag.](https://arxiv.org/abs/2111.09734)

[From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions(Young et al., TACL 2014)](https://aclanthology.org/Q14-1006) 
