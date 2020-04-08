# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 08:42:08 2020

@author: Martin
"""

from torch import empty
from math import tanh

class Module(object):
    def forward(self , *inp):
        '''
        Parameters
        ----------
        *inp : TYPE
            input.
            
        Returns
        -------
        a tensor or a tuple of tensors.
        '''
        raise  NotImplementedError
    def backward(self , *gradwrtoutput):
        '''   
        Parameters
        ----------
        *gradwrtoutput : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        '''
        raise  NotImplementedError
    def param(self):
        '''
        Returns
        -------
        list
            a  list  of  pairs,  each  composed  of  a 
            parameter  tensor, and  a  gradient  tensor
            of same size.
            This list should be empty for parameterless
            modules (e.g.  ReLU)
        '''
        return []
    

class Parameters(object):    
    def __init__(self, *size):
        self.value = empty(size)
        self.value[:] = 1 #TODO: initialize weights (xavier initialization?)
        self.grad = empty(size)
        self.grad[:] = 0
    def as_pair(self):
        return self.value, self.grad
    
class Linear(Module):
    def __init__(self, size_in, size_out):
        self.weights = Parameters(size_in, size_out)
        self.bias = Parameters(size_out)
        self.inp = empty(size_in)
        self.out = empty(size_out)
    def forward(self, inp):
        self.inp = inp
        self.out = self.weights.value @ self.inp
        return self.out
    def backward(self, gradwrtoutput):
        self.weights.grad += 0 #TODO
        self.bias.grad += 0 #TODO
        return self.weights.value @ gradwrtoutput
    def param(self):
        return [param.as_pair() for param in [self.weights, self.bias]] 
    
class ReLu(Module):
    def __init__(self, size):
        self.inp = empty(size)
        self.out = empty(size)
    def forward(self, inp):
        self.inp = inp
        self.out = max(0,self.inp)
        return self.out
    def backward(self, gradwrtoutput):
        return (self.inp>0) @ gradwrtoutput
    def param(self):
        return []

class TanH(Module):
    def __init__(self, size):
        self.inp = empty(size)
        self.out = empty(size)
    def forward(self, inp):
        self.inp = inp
        self.out = tanh(self.inp)
        return self.out
    def backward(self, gradwrtoutput):
        return (1-tanh(self.inp)**2) @ gradwrtoutput
    def param(self):
        return []
    
class Sequential(Module):
    def __init__(self, module1, module2):
        self.module1 = module1
        self.module2 = module2
    def forward(self, inp):
        return self.module2.forward(self.module1.forward(inp))
    def backward(self, gradwrtoutput):
        return self.module1.backward(self.module2.backward(inp))
    def param(self):
        return self.module1.param() + self.module2.param()
    
class LossMSE(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += (inp[k]-groundtruth[k])**2
        out /= inp.shape[1]
    def gradient(self, inp, groundtruth):
        return 2 * (inp-groundtruth) / inp.shape[1]
    
        
    
   
        
        