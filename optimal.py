import torch
import torch.nn as nn
import torch.optim as optim


model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
optimizer_l2=optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)
criterion=nn.MSELoss()


output=model(input)
loss=criterion(output,targets)

l1_lambda=1e-5
l1_norm=sum(p.abs().sum() for p in model.parameters())
loss=loss+l1_lambda*l1_norm
