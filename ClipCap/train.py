import torch
from tqdm import tqdm
from dataset import DataSet
from model import Model
from transformers import BertTokenizer, GPT2Model, AdamW, logging

def train(model, lr, epoches, loader, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    
    model.train()
    for e in range(epoches):
        ls = 0
        for i, data in tqdm(enumerate(loader)):
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(**data)
            loss = out['loss']
            ls += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 解决梯度爆炸

            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        print(e, ls / len(loader))

    model = model.cpu()
    return model
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载预训练模型
    logging.set_verbosity_error()   # 消除未使用权重的warning
    text_model = GPT2Model.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    print('预训练模型加载成功！')

    # 加载数据集
    dataset = DataSet('./data/train_emb.pt', './data/caption_validation_annotations.json', tokenizer)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 8, shuffle = True, drop_last = True)
    print('数据集加载成功！')

    # 定义模型
    model = Model(text_model, prefix_len=2, const_len=0, tokenizer=tokenizer, device=device)
    model = model.cuda() if(device == 'cuda') else model.cpu()
    print('模型定义成功！')

    # 训练
    epoches = 20
    lr = 1e-4
    model = train(model, lr, epoches, loader, device)
    print('训练完毕！')

    # 保存
    torch.save(model, './models/_clipcap_mlp_finetune.model')
    print('模型保存成功！')