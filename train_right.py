import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
from PIL import Image
import cv2
import csv
import copy

from collections import OrderedDict
from scipy import spatial
import glob

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models
import torchvision.transforms as transforms


import pickle
from joblib import dump, load

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


data_dir = '../archive'

df = pd.read_csv('./train.csv')

#train for the left part
df_right = df[df['laterality'] == 'R'].reset_index() 

df_cc_right = df_right[df_right['view'] == 'CC'].reset_index()

print(df_cc_right.iloc[0])
transform = transforms.Compose([transforms.Resize((256, 256)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

class CancerDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.patient_data = df
        self.root_dir = root_dir
        self.transform = transform


    def __getitem__(self, index):
        
        if index >= self.__len__():
            print("Error: Index out of bounds")
            return
        
        label = int(self.patient_data['cancer'].iloc[index])
        image_name = str(self.patient_data['patient_id'].iloc[index]) + '_'+ str(self.patient_data['image_id'].iloc[index])+'.png'
        path = os.path.join(self.root_dir,image_name)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        
        return image, label

    def __len__(self):
        return len(self.patient_data)

def getAUC(y_true, y_score, task):
    auc = 0
    zero = np.zeros_like(y_true)
    one = np.ones_like(y_true)
    for i in range(y_score.shape[1]):
        y_true_binary = np.where(y_true == i, one, zero)
        y_score_binary = y_score[:, i]
        try:
            auc += roc_auc_score(y_true_binary, y_score_binary)
        except ValueError:
            auc += 0.75
    return auc / y_score.shape[1]

epoch = 40
batch_size = 8
weights = 'weights_right'
lr_rate = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import timm

#model = timm.create_model('resnet34', pretrained = True, in_chans = 3,  num_classes=2).to(device)
model = timm.create_model('swinv2_tiny_window16_256', pretrained = True, img_size=256, num_classes=2).to(device)

from sklearn.model_selection import StratifiedShuffleSplit

# create an instance of the class
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_index, test_index = next(sss.split(df_cc_right, df_cc_right['cancer']))
print(len(df_cc_right.iloc[train_index]))
print(len(df_cc_right.iloc[test_index]))
df_cc_right_train = df_cc_right.iloc[train_index] 
df_cc_right_test = df_cc_right.iloc[test_index]
train_cancer_dataset = CancerDataset(df_cc_right_train, data_dir, transform)
test_cancer_dataset = CancerDataset(df_cc_right_test, data_dir, test_transform)
train_loader = DataLoader(train_cancer_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_cancer_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr_rate)




def train(model, optimizer, criterion, train_loader, device, epoch, end_epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.squeeze().long().to(device)
        #targets = targets.long().to(device)
        try:
            num_targets = len(targets)
            targets = targets.long().resize_(num_targets)
        except TypeError:
            num_targets = 1
            targets = targets.long().resize_(num_targets)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('******* Epoch [%3d/%3d], Phase: Training, Batch [%4d/%4d], loss = %.8f *******' %
                (epoch+1, end_epoch+1, batch_idx+1, len(train_loader), loss.item()))

def val(model, val_loader, device, val_auc_list, weights_folder, epoch, end_epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long().to(device)
            try:
                num_targets = len(targets)
                targets = targets.float().resize_(num_targets, 1)
            except TypeError:
                num_targets = 1
                targets = targets.float().resize_(num_targets, 1)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            #print('******* Epoch [%3d/%3d], Phase: Validation, Batch [%4d/%4d] *******' %
            #    (epoch+1, end_epoch+1, batch_idx+1, len(val_loader)))

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, 'multi-class')
        train_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }
    if end_epoch == 30:
        print('\n******* Epoch [%3d/%3d], Training: %.5f *******\n' % (epoch+1, end_epoch, auc))
        if auc >= max(train_auc_list) :
            torch.save(state, weights_folder + '/best.pt')
            print('saving the best model')
    else:
        print('\n******* Epoch [%3d/%3d], Validation AUC: %.5f *******\n' % (epoch+1, end_epoch+1, auc))

train_auc_list = []
val_auc_list = []
for i in range(epoch):
    train(model, optimizer, criterion, train_loader, device, i, 29)
    #scheduler.step()
    val(model, train_loader, device, train_auc_list, weights, i, 30 )
    val(model, val_loader, device, val_auc_list,weights, i, 29 )

model.load_state_dict(torch.load(weights + '/best.pt')['net'])
model = model.to(device)
print('start testing the model')
val(model, val_loader, device, val_auc_list, weights,i, 29 )
