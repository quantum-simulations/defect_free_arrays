# building a CNN with PyTorch
# https://www.datacamp.com/tutorial/pytorch-cnn-tutorial


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

batch_size = 60 

# MNIST data loading and transformation
train_dataset = datasets.MNIST(root = '/Users/Usuario/Documents/GitHub/defect_free_arrays/scripts/ai', download = True, train = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root = '/Users/Usuario/Documents/GitHub/defect_free_arrays/scripts/ai', download = True, train = False, transform = transforms.ToTensor())
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

#example of a random batch of images
#we define a function to show the image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #from (channel, height, width) to (height, width, channel) (matplotlib expects channel last format)
    plt.show()
#get some random training images
dataiter = iter(train_loader) #we turn the train_loader into an iterator
images, labels = next(dataiter) #we get the next batch of images and labels
#show images
imshow(torchvision.utils.make_grid(images))
#show labels
print(f"Labels: {labels}")

# Create the CNN model
class CNN(nn.Module): #class that inherits from nn.Module
    
    def __init__(self, in_channels, num_classes):
       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super().__init__() #calls the constructor of the parent class nn.Module so that we can use its methods





