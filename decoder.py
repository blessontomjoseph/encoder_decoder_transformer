import torch

def masked_multihead_self_attetion(query,key,value):
    hidden_dim,seq_dim=query.shape[-1],query.shape(-2)
    weights=torch.bmm(query*key.transpose(-1,-2))/hidden_dim
    mask=torch.tril(torch.ones(seq_dim,seq_dim)).unsqueeze(0)
    weights=torch.softmax(weights.masked_fill(mask==0,-float("inf")),dim=-1)
    scores=torch.bmm(weights,value)
    return scores


    
    
    