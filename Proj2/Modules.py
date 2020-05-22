from torch import empty, Tensor, randn, manual_seed
from math import tanh, sqrt, log, exp

#torch.set_grad_enabled(False)

class Module(object):
    '''
    Base Class for components of neural network
    '''
    
    def _init_(self):
        self.parameters = [] #initialization of list of parameters for a module 
    
    
    def forward(self , *inp):
        '''
        Parameters
        ----------
        *inp : tensor
            input.
            
        Returns
        -------
        a tensor.
        '''
        raise  NotImplementedError
    def backward(self , *gradwrtoutput):
        '''   
        Parameters
        ----------
        *gradwrtoutput : tensor
            the gradient with respect to the output.
        Returns
        -------
        a tensor.
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
    def __init__(self, *size, w, Xavier=False, man_seed=False):
        if man_seed==True :
            manual_seed(7)
        if w == True:
            #sigma=1.
            if Xavier==True:
                sigma = sqrt(2/(size[0]+size[1]))
            self.value = empty(size).normal_(mean=0,std=1)
        else :
            self.value = empty(size).normal_(mean=0,std=1)
        self.grad = empty(size)
        self.grad[:] = 0
    def as_pair(self): # returns a tuple (value, gradient)
        return self.value, self.grad

# ---------------------------- MODULES OF NEURAL NETWORK -----------------------          

class Linear(Module):
    '''
    
    Class corresponding to a hidden layer containing weights and bias defined as Parameters.
    
    
    '''
    def __init__(self, size_in, size_out, Xavier=False, man_seed=False):
        super().__init__()    
        self.weights = Parameters(size_in, size_out, w=True, Xavier=Xavier, man_seed=man_seed) # weight parameters of the layer(inherit Parameter initialization)
        self.bias = Parameters(size_out, w=False, Xavier=False, man_seed=man_seed)
        self.inp = empty(size_in)
        self.out = empty(size_out)
        self.size_in = size_in 
        self.size_out= size_out
        self.parameters = [self.weights, self.bias]
    def forward(self, inp):
        self.inp = inp
        self.out =  self.inp @ self.weights.value + self.bias.value
        return self.out
    def backward(self, gradwrtoutput):
        self.weights.grad +=   self.inp[:,None] @ gradwrtoutput[None,:]
        self.bias.grad += gradwrtoutput 
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
        self.out = Tensor([max(0,i) for i in self.inp])
        return self.out
    def backward(self, gradwrtoutput):
        return Tensor([int(i>=0) for i in self.inp]) * gradwrtoutput

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

class Softmax(Module):
    def __init__(self, *size):
        super().__init__()
        self.inp = empty(size)
        self.out = empty(size)
        self.parameters=[]
    def forward(self, inp):
        self.inp = inp
        M = self.inp.max()
        N = sum([exp(i-M) for i in self.inp])
        self.out = Tensor([exp(i-M)/N for i in self.inp])
        return self.out
    def backward(self, gradwrtoutput):
        A = -self.out[:,None]@self.out[None,:] + Tensor([[self.out[i]*int(i==j) for i in range(len(self.out))] for j in range(len(self.out))]) 
        return A @ gradwrtoutput

            
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

# define another class for large networks??? 
            
            
#---------------------------------- LOSS function --------------------------------
            
            
class LossMSE(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += (inp[k]-groundtruth[k])**2
        out /= inp.shape[0]
        return out
    def gradient(self, inp, groundtruth):
        return 2 * (inp-groundtruth) / inp.shape[0]    
    
class CrossEntropyLoss(object):
    def compute(self, inp, groundtruth):
        out = 0
        for k in range(len(inp)):
            out += -log(inp[k],2)*groundtruth[k]
        return out
    def gradient(self, inp, groundtruth):
        return Tensor([-groundtruth[k]/inp[k] for k in range(len(inp))])
    
#---------------------------------------SGD---------------------------------------

class SGD() :
    def __init__(self, params, lr, momentum=0.5):
        if lr < 0.0 :
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.params = params
        self.lr = lr
        self.momentum=momentum
        self.Vt = []
        for i in range(len(self.params)):
            self.Vt.append(self.params[i].grad.clone())
    def step(self) :
        for i in range(len(self.params)):
            self.Vt[i] = self.Vt[i]*self.momentum - self.lr*self.params[i].grad
            self.params[i].value = self.params[i].value + self.Vt[i]
            self.params[i].grad[:] = 0
