# -*- coding: utf-8 -*-

from dlc_practical_prologue import generate_pair_sets
from dataset import PairSetDataset
from models import PairSetsModel
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics

# Parameters
n = 1000
batch_size = 50
num_workers = 0
epochs = 25
rounds = 1

# History
accuracy_history_rounds = []
loss_history_rounds = []
accuracy_history_test_rounds = []
loss_history_test_rounds = []

# Visualize weights
visual = True

        
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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
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
                model.train()
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
                model.eval()
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
        
        if visual and rnd == rounds-1:            
            for m in range(4):
                model_name = ["Weight Sharing", "Weight Sharing, Aux_loss", "Nothing", "Aux_loss"][m]
                weights_1st_layer = list([module for module in models[m].modules()][3].weight.data)
                weights_filters_1st_layer = [list(weights_1st_layer[k])[0] for k in range(4)]
                figure = plt.figure(figsize=(2, 2))
                plt.title(f'{epoch+1}, "{model_name}"')
                #plt.title(f'Filters of 1st layer of 1st DigitNet after {epoch+1} epoch(s), last round, model "{model_name}"')
                plt.axis('off')
                for k in range(4):
                    ax = figure.add_subplot(2,2,k+1)
                    ax.imshow(weights_filters_1st_layer[k])
                    ax.axis('off')
                plt.show()
        
    accuracy_history_rounds.append(accuracy_history)
    loss_history_rounds.append(loss_history)
    accuracy_history_test_rounds.append(accuracy_history_test)
    loss_history_test_rounds.append(loss_history_test)
    
colors = ['red', 'blue', 'green', 'orange']
    
def plot(history_rounds, name, minus=False, log=False):
    mean = [[0 for model in range(4)] for epoch in range(epochs)]
    std = [[0 for model in range(4)] for epoch in range(epochs)]
    for epoch in range(len(history_rounds[0])):
        for model in range(4):
            mean[epoch][model] = statistics.mean([history_rounds[rnd][epoch][model] for rnd in range(rounds)])
            if rounds>2:
                std[epoch][model] = statistics.stdev([history_rounds[rnd][epoch][model] for rnd in range(rounds)])
            else :
                std[epoch][model] = 0
    for model in range(4):
        y_mean = [mean[epoch][model] for epoch in range(epochs)]
        if minus: y_mean = [n-y_mean[epoch] for epoch in range(epochs)]
        y_std = [std[epoch][model] for epoch in range(epochs)]
        down = [y_mean[i] - y_std[i] for i in range(epochs)]
        up = [y_mean[i] + y_std[i] for i in range(epochs)]
        plt.plot(range(epochs), y_mean, color=colors[model])
        if log: plt.yscale('log')
        plt.fill_between(range(epochs), up, down, color=colors[model], alpha=.3)
    plt.legend(["Weight Sharing", "Weight Sharing, Aux_loss", "", "Aux_loss"])
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.show()


for log in [True, False] :
    plot(accuracy_history_rounds, 'Train Accuracy', minus=False, log=log)
    plot(accuracy_history_rounds, 'Train Errors', minus=True, log=log)
    plot(loss_history_rounds, 'Train Loss', minus=False, log=log)
    plot(accuracy_history_test_rounds, 'Test Accuracy', minus=False, log=log)
    plot(accuracy_history_test_rounds, 'Test Errors', minus=True, log=log)
    plot(loss_history_test_rounds, 'Test Loss', minus=False, log=log)

    

