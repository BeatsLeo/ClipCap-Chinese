import torch
from transformers import GPT2LMHeadModel

class Model(torch.nn.Module):
    def __init__(self, gpt, prefix_len, const_len, tokenizer, device='cpu'):
        super().__init__()
        self.device = device
        self.prefix_len = prefix_len
        self.const_len = const_len

        self.mapping_vision = torch.nn.Sequential(torch.nn.Linear(1024, prefix_len*768))
        self.text_gen = gpt
        # 加载生成模型生成部分最后一层fc参数
        self.fc = torch.nn.Linear(768, tokenizer.vocab_size, bias=False).to(self.device)
        parameters = GPT2LMHeadModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall').to(self.device)
        self.fc.load_state_dict(parameters.lm_head.state_dict())

        self.criterion =torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, img_feature, label_ids, label_attention_mask):
        bs = img_feature.shape[0]

        # 将图像特征map到文本特征
        prefix_embeddings = self.mapping_vision(img_feature).view(bs, self.prefix_len, 768)
        label_embeddings = self.text_gen.wte(label_ids)
        
        # 文本生成
        logits = torch.concat([prefix_embeddings, label_embeddings], dim=1)
        prefix_mask = torch.ones(bs, self.prefix_len+self.const_len).to(self.device)
        attention_mask = torch.concat([prefix_mask, label_attention_mask], dim=1)   # 补齐mask
    
        logits = self.text_gen(inputs_embeds=logits, attention_mask=attention_mask).last_hidden_state
        logits = self.fc(logits)

        # 取有效值算loss
        loss_mask_head = torch.zeros(bs, self.prefix_len+self.const_len-1).to(self.device)
        loss_mask_tail = torch.zeros(bs, 1).to(self.device)
        loss_mask = torch.concat([loss_mask_head, label_attention_mask, loss_mask_tail], dim=1)
        # 计算损失
        shift_logits = logits[loss_mask == 1]
        shift_labels = label_ids[label_attention_mask == 1]

        loss = self.criterion(shift_logits, shift_labels)

        return {
            'loss': loss,
            'logits': logits
        }
    
    def cuda(self):
        self.to('cuda')
        self.device = 'cuda'
        return self

    def cpu(self):
        self.to('cpu')
        self.device = 'cpu'
        return self