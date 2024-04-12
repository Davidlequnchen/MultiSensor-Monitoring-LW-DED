'''
Created by: Chen Lequn

Some helper functions for PyTorch model training and testing. 
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchaudio
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, resample, class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

from datetime import datetime
from tqdm import tqdm
## plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.pyplot import gca
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import seaborn as sns
from itertools import cycle
import itertools


def plot_data_distribution(data, variable, title, filename, figure_size=(7, 6)):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create the countplot
    sns.countplot(x=variable, data=data, palette='Set1', saturation=0.7, edgecolor='k', linewidth=1.5, ax=ax, alpha = 0.9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    # Add percentages above each bar
    total = len(data)
    max_height = 0
    for p in ax.patches:
        max_height = max(max_height, p.get_height())
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
    
    # Extend the y-axis for better visibility of annotations
    ax.set_ylim([0, max_height * 1.2])
    
    # Set labels and title
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("Categories", fontsize=20, labelpad=12)
    ax.set_ylabel("Data volume", fontsize=20, labelpad=10)
    ax.tick_params(labelsize=15)
    
    # Add grid
    ax.grid(True, which='both', axis='y', linestyle='dotted', linewidth=0.5, alpha=0.7, color='black')
    
    # Show all four edges of the plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Show the plot
    plt.tight_layout()
    



def train_single_epoch(model, epoch, trainloader, loss_fn, optimizer, device, mode = "single_model"):
    '''
    Function for the training single epoch in the training loop
    '''
    print('\nEpoch: %d' % epoch)
    model.train() # training mode
    running_loss = 0
    train_loss = 0
    correct = 0
    total = 0

    #  create a progress bar for the training data loader
    pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if mode == "single_model":
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
        elif mode == "multi_model":
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            ## forward pass and calculate loss
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs[0].size(0)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record and update current progress
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # update the progress bar with the loss, accuracy
        pbar.set_postfix({'loss': train_loss/(batch_idx+1), 
                          'accuracy': 100.*correct/total})

    acc = 100.*correct/total
    epoch_loss = running_loss / len(trainloader.dataset)

    return model, optimizer, epoch_loss, acc


def test_single_epoch(model, epoch, testloader, loss_fn, device, mode = "single_model"):
    model.eval() # evaluation mode
    global best_acc # for updating the best accuracy so far
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        #  create a progress bar for the training data loader
        pbar = tqdm(testloader, desc=f"Epoch {epoch}") 

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if mode == "single_model":
                inputs, targets = inputs.to(device), targets.to(device)
                ## forward pass and calculate loss
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item() *inputs.size(0)
            elif mode == "multi_model":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                ## forward pass and calculate loss
                outputs = model(*inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item() *inputs[0].size(0)

            # record current progress
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # update the progress bar with the loss, accuracy
            pbar.set_postfix({'loss': test_loss/(batch_idx+1), 
                          'accuracy': 100.*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'model': model.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc
    
    epoch_loss = test_loss / len(testloader.dataset)
    return model, epoch_loss, acc



def training_loop(model, loss_fn, optimizer, train_loader, valid_loader, epochs, scheduler, device, print_every=1, mode = "single_model"):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
 
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss, train_acc = train_single_epoch(model, epoch, train_loader, loss_fn, optimizer, device, mode = mode)
        train_losses.append(train_loss)
        # validation
        model, valid_loss, valid_acc = test_single_epoch(model, epoch, valid_loader, loss_fn, device, mode = mode)
        valid_losses.append(valid_loss)

        # if epoch % print_every == (print_every - 1):
            
        #     train_acc = get_accuracy(model, train_loader, device=device)
        #     valid_acc = get_accuracy(model, valid_loader, device=device)
                
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
              f'Epoch: {epoch}\t'
              f'Train loss: {train_loss:.4f}\t'
              f'Valid loss: {valid_loss:.4f}\t'
              f'Train accuracy: {train_acc:.2f}\t'
              f'Valid accuracy: {valid_acc:.2f}')
        
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)
        scheduler.step()
    return model, optimizer, (train_losses, valid_losses, train_accuracy, valid_accuracy)


