# -*- coding: utf-8 -*-
from torch import rand, norm, Tensor, sign
from math import pi, sqrt
import statistics
from Modules import Sequential, Linear, TanH, Softmax, LossMSE, CrossEntropyLoss, SGD
import matplotlib.pyplot as plt

# Dataset
N = 1000

class Data_set(object):
    def __init__(self, N, dim=2, r=1/sqrt(2*pi), center=[.5,.5], man_seed=False, t=False) :
        if man_seed==True:
            if t == True:
                manual_seed(123)
            else :
                manual_seed(124)
        self.points=rand(size=(N,dim))
        self.distances=norm(self.points-Tensor(center), 2, dim=1)
        self.labels=(sign(r - self.distances))/2 +.5
        self.size=N
        self.groundtruth = [Tensor([1., 0.]) if self.labels[i]==0 else Tensor([0., 1.]) for i in range(self.size)]
        
test = Data_set(N,2)
trn = Data_set(N,2)
# Generates a training and a test set of 1,000 points sampled uniformly in [0,1]²
#points_train = torch.rand(size=(N,2))
#points_test = torch.rand(size=(N,2))
   
# Generates labels: label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside
#distances_train = torch.norm(points_train-torch.Tensor([[.5, .5]]), 2, dim=1)
#distances_test = torch.norm(points_test-torch.Tensor([[.5, .5]]), 2, dim=1)
#labels_train = (torch.sign(1/sqrt(2*pi) - distances_train))/2 +.5 #returns 0,1 hot labels
#labels_test = (torch.sign(1/sqrt(2*pi) - distances_test))/2 +.5


# Hyperparameters
lr = 0.001
N_iter = 10
N_epochs = 25
epochs = range(N_epochs)

# Builds a network with two input units, two output units, three hidden layers of 25 units
net = Sequential(Linear(size_in=2, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=2),
                 Softmax(2),
                )

loss = CrossEntropyLoss()
# History
train_errors_history = Tensor(N_iter,N_epochs)
train_loss_history = Tensor(N_iter,N_epochs)
test_errors_history = Tensor(N_iter,N_epochs)
test_loss_history = Tensor(N_iter,N_epochs)

# Training loop
for j in range(N_iter):
    test = Data_set(N,2)
    trn = Data_set(N,2)
    net = Sequential(Linear(size_in=2, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=2),
                 Softmax(2),
                )
    optimizer = SGD(net.parameters, lr, momentum=0.0)
    for epoch in epochs:
        # Train
        train_errors = 0
        train_loss = 0
        for i in range(trn.size):
            #Updating the gradients at first with SGD
            output = net.forward(trn.points[i])
            grad = loss.gradient(output, trn.groundtruth[i])
            net.backward(grad)
            optimizer.step()
        for i in range(trn.size):
            output = net.forward(trn.points[i])
            prediction = 0 if output[0]>output[1] else 1
            if prediction != trn.labels[i]:
                train_errors+=1
            l = loss.compute(output, trn.groundtruth[i])
            train_loss += l
        train_loss /= trn.size
        train_errors /= trn.size
            # Test
        test_errors = 0
        test_loss = 0
        for i in range(test.size):
        #print(f"point {points_train[i]}")
            output = net.forward(test.points[i])
            prediction = 0 if output[0]>output[1] else 1
            l = loss.compute(output, test.groundtruth[i])
            if prediction != test.labels[i]:
                test_errors+=1
            test_loss += l
        test_loss /= test.size
        test_errors /= test.size
        # Update history
        print (f"epoch {epoch+1}, train errors {train_errors}, test errors {test_errors}, train loss {train_loss}, test loss {test_loss}")
        train_errors_history[j][epoch] = train_errors
        train_loss_history[j][epoch] = train_loss
        test_errors_history[j][epoch] = test_errors
        test_loss_history[j][epoch] = test_loss




# Plots
# Loss
plt.plot(epochs, train_loss_history)
plt.plot(epochs, test_loss_history)
plt.legend(["Train", "Test"])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# Errors
plt.plot(epochs, train_errors_history)
plt.plot(epochs, test_errors_history)
plt.legend(["Train", "Test"])
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.show()
