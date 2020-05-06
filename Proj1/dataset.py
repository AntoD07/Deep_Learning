# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch

class Sample:
    def __init__(self, image1, image2, target, class1, class2):
        self.image1 = image1
        self.image2 = image2
        self.target = target
        self.class1 = class1
        self.class2 = class2
        
    def as_dict(self):
        dico = {'image1': self.image1,
                'image2': self.image2,
                'target': self.target,
                'class1': self.class1,
                'class2': self.class2
                }
        return dico

class PairSetDataset(torch.utils.data.Dataset):
    def __init__(self, pair_sets, mode = 'train'):     
        self.samples = []
        if mode == 'train':
            inputs, targets, classes, _, _, _ = pair_sets
        else:
            _, _, _, inputs, targets, classes = pair_sets
        for idx in tqdm(range(len(targets))):
            image1 = inputs[idx, 0, :, :].view(1, 14, 14)/256-.5
            image2 = inputs[idx, 1, :, :].view(-1, 14, 14)/256-.5
            target = targets[idx] + 0.        
            class1 = torch.FloatTensor(10).zero_()
            class1[classes[idx, 0]] = 1.
            class2 = torch.FloatTensor(10).zero_()
            class2[classes[idx, 1]] = 1.
            sample = Sample(image1, image2, target, class1, class2)
            self.samples.append(sample)
    
    def __getitem__(self, idx):
        return self.samples[idx].as_dict()
    
    def __len__(self):
        return len(self.samples)