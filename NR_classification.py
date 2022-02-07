# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:00:00 2022

@author: Liam Shanahan
"""

import numpy as np
import sys
import random as rd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as tud
# from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import CustomDataset as cd
import Net

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

np.set_printoptions(threshold=sys.maxsize)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def train(model, device, train_loader, optimizer, epoch, batch_size = 50):
  model.train()
  #loop run over data output by loader to train model
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data) #applies model to data
    target = target.squeeze(1) #removes dimension from target
    
    # print(output)
    # print(target.shape)
    loss = nn.CrossEntropyLoss()(output, target[:,1].long()) #calculates cross entropy loss
    loss.backward()
    optimizer.step() #new step in optimisation of loss
    #printing output of each batch for loss observation while model is training
    if batch_idx % batch_size == 0:
      print(
        f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
      )

def test(model, device, test_loader):
  model.eval()
  #initialisting parameters and empty lists
  test_loss = 0
  correct = 0
  probs_pred = []
  labels_actual = []
  labels_pred = []
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      target=target.squeeze(1)

      test_loss += nn.CrossEntropyLoss()(output, target[:,1].long()).item()
  
      pred = output.max(1, keepdim=True)[1] #finds index of maximum in output vector as described below
      pred_onehot = F.one_hot(pred).squeeze(1) #converts into one-hot predicted label 
      pred_bool = torch.eq(pred_onehot,target) #finds values where predicted equals target
      correct += int((pred_bool.sum().item())/2) #sums number of correct values
      #adds predicted labels, targets and probabilities to lists for later use
      labels_actual.append(target.cpu().numpy())
      probs_pred.append(output.cpu().numpy())
      labels_pred.append(pred_onehot.cpu().numpy())
 
  #printing accuracy and loss after each epoch for analysis
  test_loss /= len(test_loader.dataset)
  print(
    f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100 * correct / len(test_loader.dataset):.0f}%)\n'
  )
  
  return(labels_actual,probs_pred,labels_pred)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




"""
For a learning algorithm, it will be necesarry to have an input (X data) containing a number of $97\times97$ matrices for each energy. 
The output will be a label, either Carbon or Flourine. The model will compare its output with the actual labels and adjust accordingly to minimise loss. 

In order to quantify each label, the common 'one-hot encoding' method will be used. 
C elements will have the label `[1, 0]` and F will be `[0, 1]`. 
Tuples of this form will be the ground truth labels and the model output. 
I will have to ensure that the model outputs an array of shape `[2]`.
"""

# image data, one-hot elem labels, energy
im_dat, el_labs, en_labs = [],[],[]

for i in np.arange(130,181,5): # run over all energies
  C_dat = np.load('C_'+str(i)+'keV.npy')
  
  for n in range(np.shape(C_dat)[0]):
    im_dat.append(C_dat[n])
    el_labs.append([1,0])
    en_labs.append(i)
    
  F_dat = np.load('F_'+str(i)+'keV.npy')
  
  for n in range(np.shape(F_dat)[0]):
    im_dat.append(F_dat[n])
    el_labs.append([0,1])
    en_labs.append(i)
        
        
# C_dat = np.load('C_130keV.npy')

# for n in range(np.shape(C_dat)[0]):
#   im_dat.append(C_dat[n])
#   el_labs.append([1,0])
#   en_labs.append(130)
  
# F_dat = np.load('F_170keV.npy')

# for n in range(np.shape(F_dat)[0]):
#   im_dat.append(F_dat[n])
#   el_labs.append([0,1])
#   en_labs.append(170)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

X_train, X_test, y_train, y_test = train_test_split(im_dat, el_labs, 
                                                    test_size=0.8, random_state=42)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Training squeezing #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

xtrain_tensor = torch.from_numpy(np.array(X_train)) #converts numpy array to torch tensor
xtrain_sqz = xtrain_tensor.unsqueeze(1) #adds new dimension along axis 1
print(xtrain_sqz.shape)


ytrain_tensor = torch.from_numpy(np.array(y_train))
ytrain_sqz = ytrain_tensor.unsqueeze(1)
print(ytrain_sqz.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Testing squeezing #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

xtest_tensor = torch.from_numpy(np.array(X_test)) #converts numpy array to torch tensor
xtest_sqz = xtest_tensor.unsqueeze(1) #adds new dimension along axis 1
print(xtest_sqz.shape)

ytest_tensor = torch.from_numpy(np.array(y_test))
ytest_sqz = ytest_tensor.unsqueeze(1)
print(ytest_sqz.shape)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Custom dataset formatting #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

train_data = cd.CustomDataset(xtrain_sqz.float(), ytrain_sqz.float())
test_data = cd.CustomDataset(xtest_sqz.float(), ytest_sqz.float())


#defining dataloader class
train_loader = tud.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
train_loader_iter = iter(train_loader)

#same as above but for test data
test_loader = tud.DataLoader(dataset=test_data, batch_size=50, shuffle=True)
test_loader_iter = iter(test_loader)


device = torch.device('cuda')


model = Net.Net3_mod().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5) #stochastic gradient descent used as optimiser
optimizer = optim.Adam(model.parameters(), lr=1e-4)

initial_model = model
# print(model)
numel_list = [p.numel() for p in model.parameters()]
# print(sum(numel_list), numel_list)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


num_epochs = 5 #setting number of epochs for CNN

#running loop training and testing over previously specified number of epochs
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    act,pred,pred_labs = test(model, device, test_loader)








