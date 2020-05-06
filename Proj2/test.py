# -*- coding: utf-8 -*-
import torch
from math import sqrt, pi
from Modules import Sequential, Linear, TanH, LossMSE, SGD
import matplotlib.pyplot as plt

# Dataset
N = 1000

# Generates a training and a test set of 1,000 points sampled uniformly in [0,1]²
points_train = torch.rand(size=(N,2))
points_test = torch.rand(size=(N,2))
   
# Generates labels: label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside
distances_train = torch.norm(points_train-torch.Tensor([[.5, .5]]), 2, dim=1)
distances_test = torch.norm(points_test-torch.Tensor([[.5, .5]]), 2, dim=1)
labels_train = (torch.sign(1/sqrt(2*pi) - distances_train))/2 +.5
labels_test = (torch.sign(1/sqrt(2*pi) - distances_test))/2 +.5

# Hyperparameters
lr = 0.01
epochs = range(1, 101)

# Builds a network with two input units, two output units, three hidden layers of 25 units
net = Sequential(Linear(size_in=2, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=25),
                 TanH(25),
                 Linear(size_in=25, size_out=2),
                 TanH(25),
                 )
loss = LossMSE()
optimizer = SGD(net.parameters, lr)

# History
train_errors_history = []
train_loss_history = []
test_errors_history = []
test_loss_history = []


# Training loop
for epoch in epochs:
    # Train
    train_errors = 0
    train_loss = 0
    for i in range(N):
        #print(f"point {points_train[i]}")
        groundtruth = torch.Tensor([1., 0.]) if labels_train[i]==0 else torch.Tensor([0., 1.])
        #print(f"label {labels_train[i]} --> groundtruth {groundtruth}")
        output = net.forward(points_train[i])/2 + .5
        prediction = 0 if output[0]>output[1] else 1
        #print (f"output {output} --> prediction {prediction}")
        l = loss.compute(output, groundtruth)
        grad = loss.gradient(output, groundtruth)
        #print(f"loss {l}, gradient {grad}")
        net.backward(grad)
        if prediction != labels_train[i]:
            train_errors+=1
        train_loss += l
        optimizer.step()
    train_loss /= N
    # Test
    test_errors = 0
    test_loss = 0
    for i in range(N):
        #print(f"point {points_train[i]}")
        groundtruth = torch.Tensor([1., 0.]) if labels_test[i]==0 else torch.Tensor([0., 1.])
        output = net.forward(points_test[i])/2 + .5
        prediction = 0 if output[0]>output[1] else 1
        #print (f"output {output} --> prediction {prediction}")
        l = loss.compute(output, groundtruth)
        grad = loss.gradient(output, groundtruth)
        #print(f"loss {l}, gradient {grad}")
        net.backward(grad)
        if prediction != labels_test[i]:
            test_errors+=1
        test_loss += l
        optimizer.step()
    test_loss /= N
    # Update history
    print (f"epoch {epoch}, train errors {train_errors}, test errors {test_errors}, train loss {train_loss}, test loss {test_loss}")
    train_errors_history.append(train_errors)
    train_loss_history.append(train_loss)
    test_errors_history.append(test_errors)
    test_loss_history.append(test_loss)

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