# -*- coding: utf-8 -*-

from dlc_practical_prologue import generate_pair_sets
from dataset import PairSetDataset
from Models2 import PairSetsModel
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics

# Parameters
n = 1000
batch_size = 50
num_workers = 0 #BrokenPipeError: [Errno 32] Broken pipe ?
epochs = 25
rounds = 15

# History
accuracy_history_rounds = []
loss_history_rounds = []
accuracy_history_test_rounds = []
loss_history_test_rounds = []


# rounds
for rnd in range(rounds):
    print("round", rnd)
    # Models
    models = [PairSetsModel(weight_sharing=True, use_auxiliary_loss=False),
              PairSetsModel(weight_sharing=True, use_auxiliary_loss=True),
              PairSetsModel(weight_sharing=False, use_auxiliary_loss=False),
              PairSetsModel(weight_sharing=False, use_auxiliary_loss=True)
              ]
    
    # losses
    criterion = torch.nn.MSELoss()
    criterion_aux = torch.nn.MSELoss()
    
    # Optimizers
    optimizers = [torch.optim.Adam(models[i].parameters(), lr = .01) for i in range (4)]
    
    # Dataset
    pair_sets = generate_pair_sets(n)
    train_dataset = PairSetDataset(pair_sets, 'train')
    test_dataset = PairSetDataset(pair_sets, 'test')
    
    # DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # History
    accuracy_history = []
    loss_history = []
    accuracy_history_test = []
    loss_history_test = []
    
    # Train loop
    for epoch in tqdm(range(epochs)):    
        # Train
        epoch_loss = [0., 0., 0., 0.]
        epoch_accuracy = [0., 0., 0., 0.]
        for batch_idx, batch in enumerate(train_dataloader):
            with torch.no_grad():
                image1 = batch['image1']
                image2 = batch['image2']
                class1 = batch['class1']
                class2 = batch['class2']
                target = batch['target']
            for i in range(4):
                model = models[i]
                optimizer = optimizers[i]   
                # Forward
                prediction, digit1, digit2 = model(image1, image2)       
                #print(prediction, target)     
                loss = criterion(prediction, target)
                accuracy = ((prediction>0.5)==target).sum()
                epoch_accuracy[i]+=accuracy.item()  
                epoch_loss[i]+=loss.item()  
                if model.use_auxiliary_loss: 
                    aux_loss = criterion_aux(digit1, class1) 
                    aux_loss += criterion_aux(digit2, class2)
                    loss += aux_loss
                # Backward 
                optimizer.zero_grad() 
                loss.backward()          
                optimizer.step()
        accuracy_history.append(epoch_accuracy)
        loss_history.append(epoch_loss)
        
        # Evaluate
        epoch_loss = [0., 0., 0., 0.]
        epoch_accuracy = [0., 0., 0., 0.]
        for batch_idx, batch in enumerate(test_dataloader):
            with torch.no_grad():
                image1 = batch['image1']
                image2 = batch['image2']
                class1 = batch['class1']
                class2 = batch['class2']
                target = batch['target']
            for i in range(4):
                model = models[i] 
                # Forward
                prediction, digit1, digit2 = model(image1, image2)  
                loss = criterion(prediction, target)
                accuracy = ((prediction>0.5)==target).sum()
                epoch_accuracy[i]+=accuracy.item()  
                epoch_loss[i]+=loss.item()  
                if model.use_auxiliary_loss: 
                    aux_loss = criterion_aux(digit1, class1) 
                    aux_loss += criterion_aux(digit2, class2)
                    loss += aux_loss
        accuracy_history_test.append(epoch_accuracy)
        loss_history_test.append(epoch_loss)
        
    accuracy_history_rounds.append(accuracy_history)
    loss_history_rounds.append(loss_history)
    accuracy_history_test_rounds.append(accuracy_history_test)
    loss_history_test_rounds.append(loss_history_test)
    
    
def mean_and_std(history_rounds):
    mean = [[0 for model in range(4)] for epoch in range(epochs)]
    std = [[0 for model in range(4)] for epoch in range(epochs)]
    for epoch in range(len(history_rounds[0])):
        for model in range(4):
            mean[epoch][model] = statistics.mean([history_rounds[rnd][epoch][model] for rnd in range(rounds)])
            std[epoch][model] = statistics.stdev([history_rounds[rnd][epoch][model] for rnd in range(rounds)])
    return mean, std   

def errorfill(x, y, yerr, color="red", alpha_fill=0.3, ax=None):
    #https://tonysyu.github.io/plotting-error-bars.html#.XqlOX5ngphE
    ax = ax if ax is not None else plt.gca()
    ymin = [y[i] - yerr[i] for i in range(len(y))]
    ymax = [y[i] + yerr[i] for i in range(len(y))]
    ax.plot(x, y, color=color)
    plt.yscale('log')
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
colors = ['red', 'blue', 'green', 'orange']
 
mean, std  = mean_and_std(accuracy_history_rounds)
for model in range(4):
    errorfill(range(epochs), [mean[epoch][model] for epoch in range(epochs)], yerr=[std[epoch][model] for epoch in range(epochs)], color=colors[model])
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.show()

mean, std  = mean_and_std(loss_history_rounds)
for model in range(4):
    errorfill(range(epochs), [mean[epoch][model] for epoch in range(epochs)], yerr=[std[epoch][model] for epoch in range(epochs)], color=colors[model])
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()

mean, std  = mean_and_std(accuracy_history_test_rounds)
for model in range(4):
    errorfill(range(epochs), [mean[epoch][model] for epoch in range(epochs)], yerr=[std[epoch][model] for epoch in range(epochs)], color=colors[model])
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.show()

mean, std  = mean_and_std(loss_history_test_rounds)
for model in range(4):
    errorfill(range(epochs), [mean[epoch][model] for epoch in range(epochs)], yerr=[std[epoch][model] for epoch in range(epochs)], color=colors[model])
plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.show()