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

import matplotlib.pyplot as plt

accuracy_history = []
loss_history = []
accuracy_history_test = []
loss_history_test = []

dataset = dlc_practical_prologue.generate_pair_sets(1000)
train_input, train_target, train_classes, \
test_input, test_target, test_classes = dataset
train_target = train_target.float()
test_target = test_target.float()


nb_samples = len(train_input)

models = [Models.PairSetsModel(weight_sharing = True),
          Models.PairSetsModel(weight_sharing = True),
          Models.PairSetsModel(weight_sharing = False),
          Models.PairSetsModel(weight_sharing = False)
          ]

criterion = nn.MSELoss()
criterion_aux = nn.MSELoss()
aux_use = [False, True, False, True]

optimizers = [optim.Adam(models[i].parameters(), lr = .01) for i in range (4)]
 
for n in range(0, 1000): 
    
    #training
    epoch_loss = [0., 0., 0., 0.]
    epoch_accuracy = [0., 0., 0., 0.]
    for i in range(4):
        model = models[i]
        optimizer = optimizers[i]
        aux = aux_use[i]
        optimizer.zero_grad()
        
        p, x1_p, x2_p = model(train_input)
        p = p.view(train_target.size())
        
        loss = criterion(p, train_target)
        if not aux:
            loss.backward()
        if aux :
            
            onehot_train_classes1 = torch.FloatTensor(1000, 10).zero_().scatter_(1, train_classes[:,0,None], 1)
            onehot_train_classes2 = torch.FloatTensor(1000, 10).zero_().scatter_(1, train_classes[:,1,None], 1)
            print(x1_p[0,:].data, onehot_train_classes1[0,:])        
            print(x2_p[0,:].data, onehot_train_classes2[0,:])    
            print(p[0].data, train_target[0])        
            aux_loss = criterion_aux(x1_p, onehot_train_classes1) + criterion_aux(x2_p, onehot_train_classes2)
            (loss+aux_loss).backward()
        
        
        accuracy = ((p>0.5)==train_target).sum()
        epoch_accuracy[i]=accuracy.item()  
        epoch_loss[i]=loss.item()  
        
        optimizer.step()
        
    accuracy_history.append(epoch_accuracy)
    loss_history.append(epoch_loss)
        
    #testing
    epoch_loss = [0., 0., 0., 0.]
    epoch_accuracy = [0., 0., 0., 0.]
    for i in range(4):
        model = models[i]
        
        p, _, _ = model(test_input)
        p = p.view(test_target.size())
        
        loss = criterion(p, test_target)
        accuracy = ((p>0.5)==test_target).sum()
        
        epoch_accuracy[i]=accuracy.item()  
        epoch_loss[i]=loss.item()  
        
        print(i, n, accuracy.item(), loss.item())
        
    accuracy_history_test.append(epoch_accuracy)
    loss_history_test.append(epoch_loss)
    
plt.plot(accuracy_history)
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.show()

plt.plot(loss_history)
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()

plt.plot(accuracy_history_test)
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.show()

plt.plot(loss_history_test)
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.show()