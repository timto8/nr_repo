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
    def __init__(self,out_channels):
        super(Net3_mod, self).__init__()

        self.conv_layers = nn.Sequential( # 1x97x97
            nn.Conv2d(1, 16, kernel_size=2), # 16x96x96
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=4), # 8x24x24
            # nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(16, 128, kernel_size=5,padding=2), # 128x24x24
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=4), # 64x6x6
            # nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(128, 1024, kernel_size=5,padding=2), # 1024x6x6
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(1024),
            nn.AvgPool2d(kernel_size=2), # 1024x3x3
            # nn.MaxPool2d(kernel_size=(2,2)),
            
            # nn.Conv2d(1024, 2048, kernel_size=5,padding=2), # 2048x6x6
            # nn.LeakyReLU(),
            # # nn.ReLU(),
            # # nn.Dropout(0.2),
            # nn.BatchNorm2d(2048),
            # nn.AvgPool2d(kernel_size=2), # 2048x3x3
            # # nn.MaxPool2d(kernel_size=(2,2)),
            
            
        )
            
        self.fc_layers = nn.Sequential(
            nn.Linear(9216,1152),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(1152,256),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Linear(576,144),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(256,out_channels),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 9216) #flattens as described below
#         x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x