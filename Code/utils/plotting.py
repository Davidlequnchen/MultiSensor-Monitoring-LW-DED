import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torchvision.utils import make_grid
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


## function for automatically save the diagram/graph into the folder 
def save_fig(fig_id, IMAGE_PATH, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_data_distribution(data, variable, title, filename, IMAGE_PATH, figure_size=(7, 6)):
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
    save_fig(f"{filename}", IMAGE_PATH)
    plt.show()
    
### Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    im_ratio = cm.shape[1]/cm.shape[0]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, pad=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize = 16, 
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Ground Truth', fontsize=20, labelpad =12)
    plt.xlabel('Predicted', fontsize=20, labelpad =12)
    plt.xticks(fontsize=16,  rotation=30, ha='right')
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(orientation="vertical", pad=0.1, ticks=[0, 0.5, 1], fraction=0.045*im_ratio)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_title('Accuracy',fontsize=16, pad = 12)
    plt.tight_layout()

    
## Define function to get the confusion matrix and print out the plot as well
def conf_matrix(y_true, y_pred, classes=["Laser-off",'Defect-free','Cracks','Keyhole pores'] ):
    cm = confusion_matrix(y_true, y_pred)
    
    # convert to percentage and plot the confusion matrix
    cm_pct = cm.astype(float) / cm.sum(axis =1)[:,np.newaxis]
    
    # classes = le.classes_
    print(cm)
    plot_confusion_matrix(cm_pct, classes)


def visualize_samples(dataset, label_to_index, num_samples=4, title_fontsize=24):
    # Extract unique categories from the dataset
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_categories = list(set(labels))
    
    # Create a dictionary to map labels to indices
    category_samples = {category: [] for category in unique_categories}
    
    # Iterate through the dataset and collect indices for each category
    for i in range(len(dataset)):
        image, label = dataset[i]
        category_samples[label].append(i)
    
    # Inverse mapping from indices to labels
    index_to_label = {v: k for k, v in label_to_index.items()}
    
    # Plot samples for each category
    num_cols = 2
    num_rows = 2
    
    fig, axs = plt.subplots(len(unique_categories), 1, figsize=(8, 8 * len(unique_categories)))
    
    for i, category in enumerate(unique_categories):
        if len(category_samples[category]) < num_samples:
            print(f"Not enough samples for category: {category}")
            continue

        sampled_indices = random.sample(category_samples[category], num_samples)
        sampled_images = [dataset[idx][0] for idx in sampled_indices]
        
        image_grid = make_grid(sampled_images, nrow=num_cols)
        
        ax = axs[i] if len(unique_categories) > 1 else axs
        ax.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
        ax.set_title(f'{index_to_label[category]} Samples', fontsize=title_fontsize)
        ax.axis('off')
    
    plt.tight_layout()
    # plt.show()


# Function to evaluate the model on the test set
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())  # Get raw scores

    test_loss /= len(dataloader.dataset)
    test_acc = corrects.double() / len(dataloader.dataset)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return all_labels, all_preds, np.array(all_scores), test_loss, test_acc.item()



# Function to plot ROC curves
def plot_roc_curves(y_test, y_score, n_classes, classes):
    # Binarize the output labels for ROC
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(4, 3), dpi=300)
    widths = 2
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(widths)

    tick_width = 1.5
    plt.tick_params(direction='in', width=tick_width)

    # Plot micro and macro ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.2f})'.format(roc_auc["micro"]),
             color='red', linestyle=':', linewidth=2, alpha=0.8)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2, alpha=0.8)

    # Plot ROC curve for each class
    colors = cycle(["aqua", "darkblue", "darkorange", "red", 'green', 'silver', 'yellow', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, alpha=0.8,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.8)
    plt.grid(linestyle='--', alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    # plt.show() 
