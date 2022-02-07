# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:12:47 2022

@author: Liam Shanahan
"""
import torch.utils.data as tud

#writing class for custom dataset
class CustomDataset(tud.Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data #initialises X_data
        self.y_data = y_data #initialises y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index] #returns one training example
        
    def __len__ (self):
        return len(self.X_data) #returns length of the dataset