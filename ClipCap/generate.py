import torch

# 句子生成
def generate(img_feature, model, tokenizer, max_length, num_samples):

    def generate_loop(logits, output):
        with torch.no_grad():
            out = model.text_gen(inputs_embeds = logits).last_hidden_state    # attention_mask默认全1
            out = model.fc(out)[:, -1]

        # 在前num_samples个采样
        topk_value = torch.topk(out, num_samples).values
        topk_value = topk_value[:, -1].unsqueeze(dim=1)

        # 赋值
        out = out.masked_fill(out < topk_value, -float('inf'))

        # 根据概率采样, 无放回
        out = out.softmax(dim=1)
        out = out.multinomial(num_samples=1)
        if(output is None):
            output = out
        else:
            output = torch.cat([output, out], dim=1)
        
        #将输出编码
        out_embeddings = model.text_gen.wte(out)
        logits = torch.cat([logits, out_embeddings], dim=1)

        
        if(logits.shape[1] >= max_length + model.prefix_len + model.const_len):
            return output
        
        return generate_loop(logits, output)

    logits = model.mapping_vision(img_feature).view(1, model.prefix_len, 768)

    # 重复5遍
    output = None
    logits = logits.expand(5, model.prefix_len, 768)
    
    output = generate_loop(logits, output)

    # 处理输出格式
    out_text = []
    for i in range(5):
        text = tokenizer.decode(output[i].flatten(), add_special_tokens=False)
        text_str = ''.join(text).replace(' ','')
        i = text_str.find('[SEP]')
        out_text.append(text_str[:i]) if i != -1 else out_text.append(text_str)

    return out_text