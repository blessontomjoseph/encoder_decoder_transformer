import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as f
import transformers
from transformers import AutoModel, AutoTokenizer,AutoConfig

checkpoint='bert-base-cased'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
config=AutoConfig.from_pretrained(checkpoint)
torch.manual_seed(41)

text="get off my lane"
tokens=tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids']
config.vocab_size #vocab size of bert
config.hidden_size #hidden_size of bert which is the embedding size (768)


def scaled_dot_product_attention(query,key,values):
    scores=torch.bmm(query,key.transpose(-1,-2))/query.shape[-1]
    weights=torch.softmax(scores,dim=-1)
    return torch.bmm(weights,values)


class AttentionHead(nn.Module):
    def __init__(self,embed_dim,head_dim):
        super().__init__()
        self.q=nn.Linear(embed_dim,head_dim) #embed_dim=786,head_dim=786/no_of head you need
        self.k=nn.Linear(embed_dim,head_dim)
        self.v=nn.Linear(embed_dim,head_dim)
        
    def forward(self,hidden_state): #hidden_state is the original embedding
        attention_outputs=scaled_dot_product_attention(self.q(hidden_state),self.k(hidden_state),self.v(hidden_state))
        return attention_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self,config):  #config contains the detains for bert here(i am simply taking the config of bert but in reality if you wanna make a custom transformer dmak the conig on your owwn)
        super().__init__()
        embed_dim=config.hidden_size
        num_heads=config.num_attention_heads
        head_dim=embed_dim//num_heads
        self.heads=nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear=nn.Linear(embed_dim,embed_dim)

    def forward(self,hidden_state): #hidden_state is the original embedding
        heads=torch.cat([h(hidden_state) for h in self.heads],dim=-1)
        return self.output_linear(heads)

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear_1=nn.Linear(config.hidden_size,config.intermediate_size)# intermediate_size= a projection size
        self.linear_2=nn.Linear(config.intermediate_size,config.hidden_size)
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(config.hidden_dropout_prob) #dropout_prob taken from bert
    def forward(self,x): # x=mutihead attention output
        x=self.linear_1(x)
        x=self.gelu(x)
        x=self.linear_2(x)
        x=self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def  __init__(self,config):
        super().__init__()
        self.layer_norm_1=nn.LayerNorm(config.hidden_size)
        self.layer_norm_2=nn.LayerNorm(config.hidden_size)
        self.attention= MultiHeadAttention(config)
        self.feed_foreward=FeedForward(config)

    def forward(self,x):
        hidden_state=self.layer_norm_1(x)
        x=x+self.attention(hidden_state)
        x=x+self.feed_foreward(self.layer_norm_2(x))
        return x
    
    
class Embeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embeds=nn.Embedding(config.vocab_size,config.hidden_size)
        self.position_embeds=nn.Embedding(config.max_position_embeddings,config.hidden_size)
        self.layer_norm=nn.LayerNorm(config.hidden_size,eps=1e-12)
        self.dropout=nn.Dropout()
    def forward(self,input_ids):
        token_embeds=self.token_embeds(input_ids)
        pos_vector=torch.arange(input_ids.shape[-1],dtype=torch.long).unsqueeze(0)
        pos_embeds=self.position_embeds(pos_vector)
        embeds=token_embeds+pos_embeds
        embeds=self.layer_norm(embeds)
        embeds=self.dropout(embeds)
        return embeds

class TransformerEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embeddings=Embeddings(config)
        self.layers=nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    def forward(self,x):
        x=self.embeddings(x)
        for layer in self.layers:
            x=layer(x)
        return x