import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.utils import make_grid


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


def visualize_samples(dataset, num_samples=4, title_fontsize=24):
    # Extract unique categories from the dataset
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_categories = list(set(labels))
    
    # Create a dictionary to map labels to indices
    category_samples = {category: [] for category in unique_categories}
    
    # Iterate through the dataset and collect indices for each category
    for i in range(len(dataset)):
        image, label = dataset[i]
        category_samples[label].append(i)
    
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
        ax.set_title(f'{category} Samples', fontsize=title_fontsize)
        ax.axis('off')
    
    plt.tight_layout()
    # plt.show()