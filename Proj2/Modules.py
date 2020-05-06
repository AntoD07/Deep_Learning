from torch import empty, Tensor, FloatTensor
import math as m
import time


class Module(object):
    '''
    Base Class for components of neural network
    '''
    
    def _init_(self):
        self.author = 'ME&AD'
    
    
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
    
    '''
    Class that includes the values of the parameters of the neural network, i.e. the elements beeing during the learning procedure
    Also includes the gradients being associated to the parameters.
    An option of initialization with manual seed is provided.
    
    '''
    def __init__(self, *size,manual_seed=False):
        if manual_seed==True :
            torch.manual_seed(7)
        self.value = torch.randn(size)#TODO: initialize weights (xavier initialization?)
        self.grad = empty(size)
        self.grad[:] = 0
    def as_pair(self):
        return self.value, self.grad

# ------------------------ MODULES OF NEURAL NETWORK -------------------------           

class Linear(Module):
    '''
    
    Class corresponding to a hidden layer containing weights and bias defined as Parameters.
    
    
    '''
    def __init__(self, size_in, size_out, manual_seed=False):
        super().__init__()    
        self.weights = Parameters(size_in, size_out, manual_seed=manual_seed)
        self.bias = Parameters(size_out, manual_seed=manual_seed)
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
        self.weights.grad += grdwrtoutput.view(-1,1)@(self.inp.view(1,-1)) #TODO
        self.bias.grad += grdwrtoutput #TODO
        return self.weights.value @ gradwrtoutput
    def param(self):
        return [param.as_pair() for param in [self.weights, self.bias]] 

            
class ReLu(Module):
    def __init__(self, *size):
        super().__init__()
        self.inp = empty(size)
        self.out = empty(size)
    def forward(self, inp):
        self.inp = inp
        self.out = max(0,self.inp)
        return self.out
    def backward(self, gradwrtoutput):
        return (self.inp>0) @ gradwrtoutput
    def param(self):
        return None

class TanH(Module):
    def __init__(self, *size):
        super().__init__()
        self.inp = empty(size)
        self.out = empty(size)
    def forward(self, inp):
        self.inp = inp
        self.out = tanh(self.inp)
        return self.out
    def backward(self, gradwrtoutput):
        return (1-tanh(self.inp)**2) @ gradwrtoutput
    def param(self):
        return None

            
# -------------------------------------SEQUENTIAL--------------------------------            
            
class Sequential(Module): 
    def __init__(self, *modules) :
        super().__init__()
        self.module_list = []
        for module in modules :
            self.module_list.append(module)
        self.inp = self.module_list[0].inp
        self.out = self.module_list[-1].out
        
    def add_module(self, module):
        self.module_list.append(module)
        
    def forward(self, inp):
        self.inp = inp
        x = inp
        for i in self.module_list :
            x = self.module_list[i].forward(x)
        self.out = x
        return self.out
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        reversed_module_list = module_list.reverse()
        for i in reversed_module_list : 
            x = self.module_list[i].backward(x)
        return x
    def param(self):
        return [module.param() for module in self.module_list]
            
            
#----------------------- LOSS function -----------------------------------------
            
            
class LossMSE(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += (inp[k]-groundtruth[k])**2
        out /= inp.shape[1]
    def gradient(self, inp, groundtruth):
        return 2 * (inp-groundtruth) / inp.shape[1]
    
    
#---------------------------------------SGD---------------------------------------

class SGD() :
    def __init__(self, params, lr):
        if lr < 0.0 :
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
    def step(self) :
        for tup in self.params():
            if tup == None :
                continue
        for w_or_b,grad in tup :
            w_or_b += -self.lr*grad
