import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer,AutoConfig

checkpoint='bert-base-cased'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
sample_text="why you no love me?"
config=AutoConfig.from_pretrained(checkpoint)
config['head_dim']=config.hidden_size//config.num_attention_heads
tokens=torch.tensor(tokenizer(sample_text,add_special_tokens=False)['input_ids'],dtype=torch.long)

matrix=torch.randn(1,4,768)
a=nn.Linear(768,64)
b=nn.Linear(768,64)
c=nn.Linear(768,64)

query=a(matrix)
key=b(matrix)
value=c(matrix)


if __name__=="__main__":    
    print(query.shape)
    print(key.shape)
    print(value.shape)
    print(type(tokens))
    print(config)

        