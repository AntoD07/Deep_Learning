from torch import empty, Tensor, randn, manual_seed
from math import tanh


class Module(object):
    '''
    Base Class for components of neural network
    '''
    
    def _init_(self):
        self.parameters = []
    
    
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
            manual_seed(7)
        self.value = randn(size)#TODO: initialize weights (xavier initialization?)
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
        self.parameters = [self.weights, self.bias]
    def forward(self, inp, Xavier=False):
        self.inp = inp
        if Xavier == False : 
            self.out =  self.inp @ self.weights.value/self.size_in**.5 + self.bias.value
        else :
            self.out =  self.inp @ self.weights.value/(self.size_in+self.size_out)**.5 + self.bias.value
        return self.out
    def backward(self, gradwrtoutput):
        self.weights.grad +=   self.inp[:,None] @ gradwrtoutput[None,:]
        self.bias.grad += gradwrtoutput #TODO
        return self.weights.value @ gradwrtoutput
    def param(self):
        return [param.as_pair() for param in self.parameters] 

            
class ReLu(Module):
    def __init__(self, *size):
        super().__init__()
        self.inp = empty(size)
        self.out = empty(size)
        self.parameters=[]
    def forward(self, inp):
        self.inp = inp
        self.out = max(0,self.inp)
        return self.out
    def backward(self, gradwrtoutput):
        return (self.inp>0) @ gradwrtoutput

class TanH(Module):
    def __init__(self, *size):
        super().__init__()
        self.inp = empty(size)
        self.out = empty(size)
        self.parameters=[]
    def forward(self, inp):
        self.inp = inp
        self.out = Tensor([tanh(x) for x in self.inp])
        return self.out
    def backward(self, gradwrtoutput):
        return Tensor([(1-tanh(i)**2) for i in self.inp]) * gradwrtoutput

            
# -------------------------------------SEQUENTIAL--------------------------------            
            
class Sequential(Module): 
    def __init__(self, *modules) :
        super().__init__()
        self.module_list = []
        self.parameters=[]
        for module in modules :
            self.module_list.append(module)
            self.parameters += module.parameters
        self.inp = self.module_list[0].inp
        self.out = self.module_list[-1].out
        
    def add_module(self, module):
        self.module_list.append(module)
        
    def forward(self, inp):
        self.inp = inp
        x = inp
        for module in self.module_list :
            x = module.forward(x)
        self.out = x
        return self.out
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for module in self.module_list[::-1] : 
            x = module.backward(x)
        return x
    def param(self):
        return [param.as_pair() for param in self.parameters] 
            
            
#----------------------- LOSS function -----------------------------------------
            
            
class LossMSE(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += (inp[k]-groundtruth[k])**2
        out /= inp.shape[0]
        return out
    def gradient(self, inp, groundtruth):
        return 2 * (inp-groundtruth) / inp.shape[0]    
    
#---------------------------------------SGD---------------------------------------

class SGD() :
    def __init__(self, params, lr):
        if lr < 0.0 :
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
    def step(self) :
        #print (self.params)
        #print (self.params[0].value)
        for i in range(len(self.params)):
            self.params[i].value = self.params[i].value-self.lr*self.params[i].grad
            self.params[i].grad[:] = 0
