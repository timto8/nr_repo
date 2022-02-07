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

        self.conv_layers = nn.Sequential( # 1x97x97
            nn.Conv2d(1, 5, kernel_size=6), # 5x92x92
            # nn.MaxPool2d(kernel_size=(2,2)),
            nn.AvgPool2d(kernel_size=2), # 5x46x46
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(5, 10, kernel_size=5), # 10x42x42
            nn.AvgPool2d(kernel_size=2), # 10x21x21
            # nn.MaxPool2d(kernel_size=(2,2)),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(10, 50, kernel_size=6), # 50x16x16
            nn.AvgPool2d(kernel_size=2), # 50x8x8
            # nn.MaxPool2d(kernel_size=(2,2)),
            # nn.ReLU(),
            nn.LeakyReLU(),
        )
            
        self.fc_layers = nn.Sequential(
            nn.Linear(3200,1000),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(1000,300),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(300,80),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(80,2),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 3200) #flattens as described below
#         x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
      
      
      
class Net3_mod(nn.Module):
    def __init__(self):
        super(Net3_mod, self).__init__()

        self.conv_layers = nn.Sequential( # 1x97x97
            nn.Conv2d(1, 8, kernel_size=2), # 8x96x96
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(kernel_size=4), # 8x24x24
            # nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(8, 64, kernel_size=3,padding=1), # 64x24x24
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=4), # 10x6x6
            # nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(64, 256, kernel_size=3,padding=1), # 256x6x6
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=2), # 256x3x3
            # nn.MaxPool2d(kernel_size=(2,2)),
        )
            
        self.fc_layers = nn.Sequential(
            nn.Linear(2304,288),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(288,72),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Linear(300,80),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(72,2),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 2304) #flattens as described below
#         x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x