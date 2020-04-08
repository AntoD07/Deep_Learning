# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:26:14 2020

@author: Martin
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import dlc_practical_prologue
import Models

dataset = dlc_practical_prologue.generate_pair_sets(1000)
train_input, train_target, train_classes, \
test_input, test_target, test_classes = dataset
train_target = train_target.float()
test_target = train_target.float()

nb_samples = len(train_input)

models = [Models.PairSetsModel(weight_sharing = True),
          Models.PairSetsModel(weight_sharing = True),
          Models.PairSetsModel(weight_sharing = False),
          Models.PairSetsModel(weight_sharing = False)
          ]
mse1 = nn.MSELoss()
loss1 = lambda pred, target : mse1(pred,target)
mse2 = nn.MSELoss()
loss1 = lambda pred, target : mse1(pred,target)
losses = [
    ]

model_ws_naux = Models.PairSetsModel(weight_sharing = True)
criterion_ws_naux = nn.MSELoss()
optimizer_ws_naux = optim.Adam(model_ws_naux.parameters(), lr=0.01)

for n in range(0, 25):    
    p_ws_naux, _, _ = model_ws_naux(train_input)
    p_ws_naux = p_ws_naux.view(train_target.size())
    loss_ws_naux = criterion_ws_naux(p_ws_naux, train_target)
    loss_ws_naux.backward()
    optimizer_ws_naux.step()
    
    print(n, loss_ws_naux.item())
    