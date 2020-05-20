# -*- coding: utf-8 -*-
from torch import rand, norm, Tensor, sign, manual_seed
from math import pi, sqrt
import statistics
from Modules import Sequential, Linear, TanH, Softmax, LossMSE, CrossEntropyLoss, SGD
import matplotlib.pyplot as plt

# Dataset
class Dataset(object):
    def __init__(self, N, dim=2, r=1/sqrt(2*pi), center=[.5,.5], man_seed=False, test=False) :
        if man_seed==True:
            if test == True:
                manual_seed(123)
            else :
                manual_seed(124)
        # Generates a set of 1,000 points sampled uniformly in [0,1]^dim
        self.points=rand(size=(N,dim))
        # Generates labels: label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside
        self.distances=norm(self.points-Tensor(center), 2, dim=1)
        self.labels=(sign(r - self.distances))/2 +.5
        self.size=N
        self.groundtruth = [Tensor([1., 0.]) if self.labels[i]==0 else Tensor([0., 1.]) for i in range(self.size)]
  

# Hyperparameters
N = 1000
lr = 0.001
rounds = 15
N_epochs = 25
epochs = range(N_epochs)

# History
train_errors_history_rounds = []
train_loss_history_rounds = []
test_errors_history_rounds = []
test_loss_history_rounds = []

for rnd in range(rounds):
    print(f"round {rnd+1}")
    
    # Generate train and test sets
    trn = Dataset(N,2)
    test = Dataset(N,2)
    
    # Generate network
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
    loss = LossMSE()    #CrossEntropyLoss()
    
    # History
    train_errors_history = []
    train_loss_history = []
    test_errors_history = []
    test_loss_history = []
    
    # Training loop
    for epoch in epochs:
        # Train with SGD
        train_errors = 0
        train_loss = 0
        for i in range(trn.size):
            # Forward
            output = net.forward(trn.points[i])
            # Backward
            grad = loss.gradient(output, trn.groundtruth[i])
            net.backward(grad)
            optimizer.step()
            
        # Compute train metrics
        for i in range(trn.size):
            output = net.forward(trn.points[i])
            prediction = 0 if output[0]>output[1] else 1
            if prediction != trn.labels[i]:
                train_errors+=1
            l = loss.compute(output, trn.groundtruth[i])
            train_loss += l.item()
        train_loss /= trn.size
        train_errors /= trn.size
        
        # Compute test metrics
        test_errors = 0
        test_loss = 0
        for i in range(test.size):
            output = net.forward(test.points[i])
            prediction = 0 if output[0]>output[1] else 1
            l = loss.compute(output, test.groundtruth[i])
            if prediction != test.labels[i]:
                test_errors+=1
            test_loss += l.item()
        test_loss /= test.size
        test_errors /= test.size
        
        # Update history
        print (f"epoch {epoch+1}, train errors {train_errors}, test errors {test_errors}, train loss {train_loss}, test loss {test_loss}")
        train_errors_history.append(train_errors)
        train_loss_history.append(train_loss)
        test_errors_history.append(test_errors)
        test_loss_history.append(test_loss)

    # Update history
    train_errors_history_rounds.append(train_errors_history)
    train_loss_history_rounds.append(train_loss_history)
    test_errors_history_rounds.append(test_errors_history)
    test_loss_history_rounds.append(test_loss_history)

# Final plots
plots = [
    ["Error rate", train_errors_history_rounds, test_errors_history_rounds],
    ["Loss", train_loss_history_rounds, test_loss_history_rounds]
    ]
for plot in plots : 
    name, train_hist, test_hist = plot
    for hist in [train_hist, test_hist]:
        # Compute mean and std based on the different rounds
        mean = [0 for epoch in epochs]
        std = [0 for epoch in epochs]
        for epoch in epochs:
            mean[epoch] = statistics.mean([hist[rnd][epoch] for rnd in range(rounds)])
            if rounds>2:
                std[epoch] = statistics.stdev([hist[rnd][epoch] for rnd in range(rounds)])
            else :
                std[epoch] = 0  
        # Plot the mean
        plt.plot(epochs, mean)
        # Plot the std      
        down = [mean[i] - std[i] for i in epochs]
        up = [mean[i] + std[i] for i in epochs]
        plt.fill_between(epochs, up, down, alpha=.3)
    # Legend and axes
    plt.legend(["Train", "Test"])
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.yticks([0.05*k for k in range(10)])
    plt.show()
