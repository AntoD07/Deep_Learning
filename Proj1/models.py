# -*- coding: utf-8 -*-

import torch
from torch import nn   

class DigitNet(nn.Module):
    
    def __init__(self):
        super(DigitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.BatchNorm2d(4),
            #nn.Dropout2d(),
            nn.PReLU(),
            nn.Conv2d(4, 8, 3),
            nn.BatchNorm2d(8),
            #nn.Dropout2d(),
            nn.PReLU(),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            #nn.Dropout2d(),
            nn.PReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            #nn.Dropout(),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            #nn.Dropout(),
            nn.PReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        features = self.features(x)
        features = features.view(-1, 16 * 6 * 6)
        digit = self.fc(features)
        return digit
    

class PairSetsModel(nn.Module):

    def __init__(self, weight_sharing = True, use_auxiliary_loss = True):
        super(PairSetsModel, self).__init__()        
        self.weight_sharing = weight_sharing
        self.use_auxiliary_loss = use_auxiliary_loss        
        self.net1 = DigitNet()
        self.net2 = self.net1 if weight_sharing else DigitNet()        
        self.fc = nn.Linear(100, 1)

    def forward(self, image1, image2):
        digit1 = self.net1(image1)
        digit2 = self.net2(image2)
        digits = torch.bmm(digit1[:,:,None], digit2[:,None,:])
        prediction = self.fc(digits.view(-1, 100)).view(-1)       
        return prediction, digit1, digit2