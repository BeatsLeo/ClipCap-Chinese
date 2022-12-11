import cv2
import torch
import argparse
from model import Model
from generate import generate
import matplotlib.pyplot as plt
from transformers import BertTokenizer, CLIPFeatureExtractor, CLIPVisionModel, logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help = 'the path of model located', required=True)
    parser.add_argument('-i', help = 'the path of image located', required=True)
    args = parser.parse_args()


    logging.set_verbosity_error()   # 消除未使用权重的warning
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    model = torch.load(args.m)

    model.eval()
    # 导入图片
    img = cv2.cvtColor(cv2.imread(args.i), cv2.COLOR_BGR2RGB)
    print('你输入的图片: ')
    plt.imshow(img)
    plt.show()
    img = torch.from_numpy(img.transpose(2,0,1))
    
    # 图片预处理
    pixel_values = feature_extractor(img, return_tensors='pt')['pixel_values']
    img_feature = vision_model(pixel_values.to('cpu')).pooler_output.view(1, -1)

    # 生成文字
    print('生成的文字: ')
    model = model.cpu()
    out_text = generate(img_feature.to('cpu'), model, tokenizer, max_length=30, num_samples=20)
    
    for i, s in enumerate(out_text):
        print('第{}句：{}。'.format(i+1, s))   