
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torch.nn.functional import cross_entropy
from torch.nn.functional import one_hot

import torchio as tio

import tifffile as tiff

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
class NavigationDataset(Dataset):
    def __init__(self, folder_path, target_path, classes):
        """
        Args:
            folder_path (string): path to folder containing tif files
            target_path (string): path to folder containing target csv files
            classes (list): list of degrees (ie. [50,11,170,230,290,350])
        """
        super(NavigationDataset, self).__init__()
        self.data, self.target = self.process_input(folder_path, target_path, classes)
        self.augmentation=None #control by train() and evalu() mode switch
        self.transform = tio.transforms.Compose([
            tio.RandomAffine(
                scales=0,
                degrees=5,
                translation=(3,3,1),
                isotropic=True
            ),
            tio.RandomNoise(std=0.01)

            ])


    def process_input(self, folder_path, target_path, classes):
        """
        Process the input data and target labels into a format that can be used by the dataloader
        """
        #imaging data
        data=np.array([])
        for filename in os.listdir(folder_path):
            if filename.endswith(".tif"):
                if not data.any():
                    data=tiff.imread(os.path.join(folder_path, filename))
                else: 
                    data=np.concatenate((data,tiff.imread(os.path.join(folder_path, filename))),axis=0)
        data=torch.from_numpy(data).float().unsqueeze(1) #add channel dimension
        #labels
        target=np.array([])
        for filename in os.listdir(target_path):
            if filename.endswith(".csv"):
                if not target.any():
                    target=pd.read_csv(os.path.join(target_path, filename))
                    target=target['AverageOrientation'].to_numpy()
                   
                else: 
                    target=np.concatenate((target,pd.read_csv(os.path.join(target_path, filename))["AverageOrientation"].to_numpy()),axis=0)  
        # for i in range(len(target)):
        #     if target[i]<=classes[0] or target[i]>classes[-1]:
        #         target[i]=0
        #     elif target[i]<=classes[1]:
        #         target[i]=1
        #     elif target[i]<=classes[2]:
        #         target[i]=2
        #     elif target[i]<=classes[3]:
        #         target[i]=3
        #     elif target[i]<=classes[4]:
        #         target[i]=4
        #     else:
        #         target[i]=5
        for i in range(len(target)):
            # print(target[i])
            t=target[i]
            for element in classes:
                # print(target[i], element, (target[i]-element))
                if np.absolute(t-element)<20: #5
                  target[i]=1
                else:
                  target[i]=0
        # print(np.min(target),np.max(target))
        

        target=one_hot(torch.from_numpy(target).long(),num_classes=2).float()
       
        #print(torch.sum(target,0))
        # print(target.shape)
        return data, target
                
    def train(self):
        self.augmentation=True
    def eval(self):   
        self.augmentation=False
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        #arr_tensor = torch.from_numpy(arr).float()
        #normalization:
        arr=(arr-torch.min(arr))/(torch.max(arr)-torch.min(arr))

        #augmentation:
        if self.transform:
            arr = self.transform(arr)
        
        #crop: voxels that have value: 57 66 48 61 1 8 -> 55 70 45 65 0 10
        #arr_tensor=arr_tensor[55:70,45:65,0:10]
        arr=arr[:,55:70,45:65,0:10]
        
        return arr, self.target[idx]
