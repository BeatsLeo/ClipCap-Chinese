import os
import torch
from tqdm import tqdm    
from transformers import CLIPFeatureExtractor, CLIPVisionModel, logging
from torchvision import datasets, transforms


def process_image(image_path, output_path, feature_extractor, vision_model):
    lst = os.listdir(image_path)
    root, _ = os.path.split(image_path)
    data = {}
    cache = []
    print(output_path)
    # 读取图片
    data_transfrom = transforms.Compose([transforms.PILToTensor()])
    img_dataset = datasets.ImageFolder(root, transform=data_transfrom)

    # 分批处理图片
    for i, img in tqdm(enumerate(img_dataset)):
        with torch.no_grad():
            img = feature_extractor(img[0], return_tensors='pt')['pixel_values']
            img = img.to(device)
            img_feature = vision_model(img).pooler_output.view(1, -1).cpu()
            
        data[lst[i]] = img_feature

    # 合并批次
    torch.save(data, output_path)

    return data

def train_test_split(data_path, test_rate, output_path):
    dataset = torch.load(data_path)
    l = int(len(dataset) * test_rate)

    test_keys = list(dataset.keys())[:l]
    train_keys = list(dataset.keys())[l:]
    test_emb, train_emb = {}, {}

    for key in test_keys:
        test_emb[key] = dataset[key]
    for key in train_keys:
        train_emb[key] = dataset[key]

    torch.save(train_emb, output_path + 'train_emb.pt')
    torch.save(test_emb, output_path + 'test_emb.pt')
    return train_emb, test_emb


if __name__ == '__main__':

    root = './images/im_cap/caption_validation_images'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.set_verbosity_error()   # 消除未使用权重的warning
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    data = process_image(root, './test_emb.pt', feature_extractor, vision_model)