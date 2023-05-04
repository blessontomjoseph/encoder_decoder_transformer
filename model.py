# english to german translation softa setting
# say
# enc_emb-> english
# dec_emb-> german


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
config.hidden_size 


def scaled_dot_product_attention(query,key,values,masked=False):
    scores=torch.bmm(query,key.transpose(-1,-2))/query.shape[-1]
    if masked:
        hidden_dim,seq_dim=query.shape[-1],query.shape[-2]
        mask=torch.tril(torch.ones(seq_dim,seq_dim)).unsqueeze(0)
        weights=torch.softmax(weights.masked_fill(mask==0,-float("inf")),dim=-1)
    
    else:
        weights=torch.softmax(scores,dim=-1)
        
    return torch.bmm(weights,values)


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

class AttentionHead(nn.Module):
    "this can be used for encoder muti self or decoder multi masked self att"
    def __init__(self,config,model=None):
        super().__init__()
        self.q=nn.Linear(config.hidden_size,config.head_dim) #embed_dim=786,head_dim=786/no_of head you need
        self.k=nn.Linear(config.hidden_size,config.head_dim)
        self.v=nn.Linear(config.hidden_size,config.head_dim)
        self.model=model
        
    def forward(self,hidden_state): #hidden_state is the original embedding
        if self.model=='encoder':
            attention_outputs=scaled_dot_product_attention(self.q(hidden_state),self.k(hidden_state),self.v(hidden_state),masked=False)
            return {'attention':attention_outputs,'k':self.k,'v':self.v}
        elif self.model=='decoder':
            attention_outputs=scaled_dot_product_attention(self.q(hidden_state),self.k(hidden_state),self.v(hidden_state),masked=True)
            return {'attention':attention_outputs,'q':self.q}




class MultiHeadAttention(nn.Module):
    def __init__(self,config,model=None):  #config contains the detains for bert here(i am simply taking the config of bert but in reality if you wanna make a custom transformer dmak the conig on your owwn)
        super().__init__()
        self.model=model
        self.heads=nn.ModuleList([AttentionHead(config,model=model) for _ in range(config.num_heads)])
        self.output_linear=nn.Linear(config.hidden_size,config.hidden_size)

    def forward(self,hidden_state): #hidden_state is the original embedding
        if self.model=='encoder'
            encoder_keys=[]
            encoder_values=[]
            attentions=[]
            for head in self.heads:
                out=head(hidden_state)
                encoder_keys.append(out['k'])
                encoder_values.append(out['v'])
                attentions.append(out['attention'])
            return {'attention':torch.cat(attentions,dim=-1),'k':encoder_keys,'v':encoder_values}
        elif self.model=='decoder':
            decoder_queries=[]
            attentions=[]
            for head in self.heads:
                out=head(hidden_state)
                decoder_queries.append(out['q'])
                attentions.append(out['attention'])
            return {'attention':torch.cat(attentions,dim=-1),'q':decoder_queries}
        
            
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
        self.attention= MultiHeadAttention(config,model='encoder')
        self.feed_foreward=FeedForward(config)

    def forward(self,x): #embeddings from english
        hidden_state=self.layer_norm_1(x)
        enc_attention=self.attention(hidden_state)
        x=x+enc_attention['attention']
        x=x+self.feed_foreward(self.layer_norm_2(x))
        return x,enc_attention['k'],enc_attention['v']
    
    
class TransformerDecoderLayer(nn.Module):
    def  __init__(self,config):
        super().__init__()
        self.layer_norm_1=nn.LayerNorm(config.hidden_size)
        self.layer_norm_2=nn.LayerNorm(config.hidden_size)
        self.attention= MultiHeadAttention(config,model='decoder')
        self.feed_foreward=FeedForward(config)

    def forward(self,x): #enbedding from say german
        hidden_state=self.layer_norm_1(x)
        dec_attention=self.attention(hidden_state)
        x=x+dec_attention['attention']
        x=x+self.feed_foreward(self.layer_norm_2(x))
        return x,dec_attention['q']

        

class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder_embeddings=Embeddings(config) #sometimes config can be defferent for enc and dec embeddings max_len n stuff be aware
        self.decoder_embeddings=Embeddings(config)
        self.encoder_blocks=nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.decoder_blocks=nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self,encoder_input_ids,decoder_input_ids):
        encoder_input=self.encoder_embeddings(encoder_input_ids)
        decoder_input=self.decoder_embeddings(decoder_input_ids)
        for encoder_block,decoder_block in zip(self.encoder_blocks,self.decoder_blocks):
            encoder_input,keys,values=encoder_block(encoder_input)
            masked_multihead_attention,queries=decoder_block(decoder_input)
            encoder_decoder_attention=torch.cat([scaled_dot_product_attention(queries[i],keys[i],values[i]) for i in range config.num_attention_heads],dim=-1)
            decoder_input=masked_multihead_attention+encoder_decoder_attention
        return decoder_input