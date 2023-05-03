import torch
import torch.nn as nn
matrix=torch.randn(1,4,768)
a=nn.Linear(768,64)
b=nn.Linear(768,64)
c=nn.Linear(768,64)
query=a(matrix)
key=b(matrix)
value=c(matrix)