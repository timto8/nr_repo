# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:00:00 2022

@author: Liam Shanahan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

params = params = {
   'axes.labelsize': 21,
   'font.size': 16,
   'font.family': 'sans-serif', 
   'font.serif': 'Arial', 
   'legend.fontsize': 18,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18, 
   'axes.labelpad': 15,
   
   'figure.figsize': [10,8], # value in inches based on dpi of monitor
   'figure.dpi': 105.5, # My monitor has a dpi of around 105.5 px/inch
   
   'axes.grid': False,
   'grid.linestyle': '-',
   'grid.alpha': 0.25,
   'axes.linewidth': 1,
   'figure.constrained_layout.use': True,
   
   # Using Paul Tol's notes:
   'axes.prop_cycle': 
      mpl.cycler(color=['#4477aa', # blue
                        '#ee6677', # red/pink
                        '#228833', # green
                        '#aa3377', # purple
                        '#66ccee', # cyan
                        '#ccbb44', # yellow
                        '#bbbbbb', # grey
                        ]),
      
      
   'lines.linewidth': 2.5,
   
   'image.cmap': 'jet',
} 

plt.rcParams.update(params)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def train(model, device, train_loader, optimizer, epoch, batch_size = 50):
  model.train()
  #loop run over data output by loader to train model
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data) #applies model to data
    # print(target.shape)
    target = target.squeeze(1) #removes dimension from target
    # print(target.shape)
    # print(output)
    # print(target.shape)
    # loss = nn.CrossEntropyLoss()(output, target[:,1].long()) #calculates cross entropy loss
    
    loss = nn.CrossEntropyLoss()(output, target.long())
    # print(loss)
    model.zero_grad()
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
      # print(output.shape)
      # test_loss += nn.CrossEntropyLoss()(output, target[:,1].long()).item()
      # print(output)
      test_loss += nn.CrossEntropyLoss()(output, target.long()).item()
      
      # pred = output.max(1, keepdim=True)[1] #finds index of maximum in output vector as described below
      
      pred = output.max(1, keepdim=False)[1]#torch.round(output)
      
      # print(output)
      # pred_onehot = F.one_hot(pred).squeeze(1) #converts into one-hot predicted label 
      # print(pred_onehot)
      # print(test_loss)
      # pred_bool = torch.eq(pred_onehot,target) #finds values where predicted equals target
      pred_bool = torch.eq(pred,target)
      # print(pred_bool)
      # correct += int((pred_bool.sum().item())/2) #sums number of correct values
      correct += pred_bool.sum().item()
      #adds predicted labels, targets and probabilities to lists for later use
      labels_actual.append(target.cpu().numpy())
      probs_pred.append(output.cpu().numpy())
      # labels_pred.append(pred_onehot.cpu().numpy())
 
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
# im_dat, el_labs, en_labs = [],[],[]
im_dat, el_labs = [],[]

energies = np.arange(150,196,5)

for i in range(len(energies)): # run over all energies
  E = energies[i]

  C_dat = np.load('C_'+str(E)+'keV.npy')
  
  for n in range(np.shape(C_dat)[0]):
    im_dat.append(C_dat[n])
    # el_labs.append([1,0])
    el_labs.append(i)
    
    # en_labs.append(i)
    
  F_dat = np.load('F_'+str(E)+'keV.npy')
  
  for n in range(np.shape(F_dat)[0]):
    im_dat.append(F_dat[n])
    # el_labs.append([0,1])
    el_labs.append(i+len(energies)) # offset fluorines
    # en_labs.append(i)
        
# multiclass classifying [carbon energies... fluorine energies]
# Sum up probabilities of C's and F's to determine whether carbon or fluorine
# Can then find the peak of the distribution to extract the energy?
# Or just take most probable energy?        

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
                                                    test_size=0.2, random_state=42)
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



#%%


plt.figure(figsize=(12,10))
i = 37
E_spec_labels = [f"C {i} keV" for i in range(150,196,5)]+[f"F{i} keV" for i in range(150,196,5)]
E_labels = np.array(list(range(150,196,5)) + list(range(150,196,5)))
plt.bar(
  E_spec_labels,
  pred[-1][i],
  edgecolor='k',
  lw=2
)
plt.xticks(rotation="vertical")
plt.ylabel("Weight")
tot_prob = np.sum(pred[-1][i])
C_prob = np.sum(pred[-1][i,:10])
F_prob = np.sum(pred[-1][i,10:])
C_E = np.sum(np.arange(10)*pred[-1][i,:10]/C_prob)
F_E = np.sum(np.arange(10,20)*pred[-1][i,10:]/F_prob)
pred_elem = "C" if C_prob > F_prob else "F"
pred_E = C_E if C_prob > F_prob else F_E
act_elem  = "C" if act[-1][i] < 10 else "F"

final_energy = np.round(E_labels[int(pred_E)] + 5 * (pred_E % int(pred_E)),1)

plt.title(f"Actual: {E_spec_labels[int(act[-1][i])]}\nPredicted: {pred_elem} {final_energy} keV",fontsize=20)
plt.tight_layout()


