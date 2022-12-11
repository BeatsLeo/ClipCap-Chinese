import json
import torch
import numpy as np

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, cap_path, tokenizer):
        dataset = torch.load(data_path)
        caption = json.load(open(cap_path))
        self.data = {'img_feature': [], 'label': []}

        for cap in caption:
            if(dataset.get(cap['image_id']) is None):
                continue
            self.data['img_feature'].append(dataset[cap['image_id']])
            label = tokenizer.batch_encode_plus(cap['caption'], return_tensors='pt', return_token_type_ids=False, add_special_tokens=True, padding='max_length', max_length=32, truncation=True)
            self.data['label'].append(label)

        self.data['img_feature'] = torch.concat(self.data['img_feature'])

    def __getitem__(self, idx):
        # 随机选择与一个caption
        label_id = np.random.randint(0, 5)

        img_feature = self.data['img_feature'][idx]
        label_ids = self.data['label'][idx]['input_ids'][label_id, 1:]
        label_attention_mask = self.data['label'][idx]['attention_mask'][label_id, 1:]

        return {'img_feature': img_feature, 'label_ids': label_ids, 'label_attention_mask': label_attention_mask}
    
    def __len__(self):
        return self.data['img_feature'].shape[0]