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
        self.value = torch.randn(size)#TODO: initialize weights (xavier initialization?)
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
        self.size_in = size_in
        self.size_out= size_out
    def forward(self, inp, Xavier=False):
        self.inp = inp
        if Xavier == False : 
            self.out =  self.inp @ self.weights.value/self.size_in**.5 + self.bias.value
        else :
            self.out =  self.inp @ self.weights.value/(self.size_in+self.size_out)**.5 + self.bias.value
        return self.out
    def backward(self, gradwrtoutput):
        self.weights.grad += 0 #TODO
        self.bias.grad += 0 #TODO
        return self.weights.value @ gradwrtoutput
    def param(self):
        return [param.as_pair() for param in [self.weights, self.bias]] 
    
class ReLu(Module):
    def __init__(self, *size):
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
    def __init__(self, *size):
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
    def __init__(self, *module_list):
        if len(module_list) == 1 :
            self.module_list = module_list
        else :
            self.module_list = []
            for module in module_list :
                self.module_list.append(module)
        self.inp = module_list[0].inp
        self.out = module_list[-1].out
    def forward(self, inp):
        self.inp = inp
        x = inp
        for i in self.module_list :
            x = self.module_list[i].forward(x)
        self.out = x
        return self.out
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for i in module_list : 
            x = self.module_list[i].backward(x)
        return x
    def param(self):
        return [module.param() for module in self.module_list]
    
class LossMSE(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += (inp[k]-groundtruth[k])**2
        out /= inp.shape[1]
    def gradient(self, inp, groundtruth):
        return 2 * (inp-groundtruth) / inp.shape[1]
        
