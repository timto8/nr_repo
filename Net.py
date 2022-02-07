# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:30:12 2022

@author: Liam Shanahan
"""
from torch import nn

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.ReLU(),
        )
            
        self.fc_layers = nn.Sequential(
            nn.Linear(810, 80),
            nn.ReLU(),
            nn.Linear(80,2),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 810) #flattens as described below
#         x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
      
      

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(6,6)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=(5,5)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.Conv2d(10, 50, kernel_size=(6,6)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
        )
            
        self.fc_layers = nn.Sequential(
            nn.Linear(3200,1000),
            nn.ReLU(),
            nn.Linear(1000,300),
            nn.ReLU(),
            nn.Linear(300,80),
            nn.ReLU(),
            nn.Linear(80,2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 3200) #flattens as described below
#         x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x