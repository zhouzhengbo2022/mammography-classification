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
df_left = df[df['laterality'] == 'L'].reset_index(drop=True) 

df_cc_left = df_left[df_left['view'] == 'CC'].reset_index(drop=True)

df_cc_left_1 = df_cc_left[df_cc_left['cancer'] == 1].reset_index(drop=True)

df_cc_left_0 = df_cc_left[df_cc_left['cancer'] == 0].reset_index(drop=True)

print(df_cc_left_0.iloc[0])
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
        self.test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    
    def __getlabel__(self, index):
        label = int(self.patinet_data['cancer'].iloc[index])
        return label
    
    def __getitem__(self, index):
        
        if index >= self.__len__():
            print("Error: Index out of bounds")
            return
        
        label = int(self.patient_data['cancer'].iloc[index])
        image_name = str(self.patient_data['patient_id'].iloc[index]) + '_'+ str(self.patient_data['image_id'].iloc[index])+'.png'
        path = os.path.join(self.root_dir,image_name)
        image = Image.open(path).convert('RGB')
        if label == 1:
            image = self.transform(image)
        else:
            image = self.test_transform(image)

        
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
        #print(y_true_binary, y_score_binary)
        try:
            auc += roc_auc_score(y_true_binary, y_score_binary)
        except ValueError:
            auc += 0.75
    return auc / y_score.shape[1]

def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0
    pred = np.zeros_like(labels)
    for i in range(pred.shape[0]):
        pred[i] = np.argmax(predictions[i])
    for idx in range(len(labels)):
        prediction = min(max(pred[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
            cfp += 1 - prediction
        else:
            cfp +=  prediction

    print(ctp, cfp, y_true_count)
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

epoch = 30
batch_size = 8
weights = 'weights'
lr_rate = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import timm

#model = timm.create_model('resnet34', pretrained = True, in_chans = 3,  num_classes=2).to(device)
model = timm.create_model('swinv2_tiny_window16_256', pretrained = True, img_size=256, num_classes=2).to(device)

from sklearn.model_selection import StratifiedShuffleSplit

# create an instance of the class
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
#train_index, test_index = next(sss.split(df_cc_left, df_cc_left['cancer']))
#print(len(df_cc_left.iloc[train_index]))
#print(len(df_cc_left.iloc[test_index]))
#df_cc_left_train = df_cc_left.iloc[train_index] 
#df_cc_left_test = df_cc_left.iloc[test_index]
#train_cancer_dataset = CancerDataset(df_cc_left_train, data_dir, transform)
#test_cancer_dataset = CancerDataset(df_cc_left_test, data_dir, test_transform)
from torch.utils.data import ConcatDataset
cancer = CancerDataset(df_cc_left_1, data_dir, test_transform)
print(len(cancer))
for i in range(100):
    cancer1 = CancerDataset(df_cc_left_1, data_dir, transform)
    cancer = ConcatDataset([cancer, cancer1])
print(len(cancer))
control = CancerDataset(df_cc_left_0, data_dir, test_transform)
dataset_all = ConcatDataset([cancer, control])
label = [dataset_all.__getitem__(i)[1] for i in range(len(dataset_all))]
for train_index, val_index in sss.split(dataset_all, label):
    train_cancer_dataset = torch.utils.data.Subset(dataset_all, train_index)
    val_cancer_dataset = torch.utils.data.Subset(dataset_all, val_index)

train_loader = DataLoader(train_cancer_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_cancer_dataset, batch_size=batch_size, shuffle=True)
class_weights = torch.Tensor([0.4, 1.2])
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
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
        pf = pfbeta(y_true, y_score, 1)
        train_pf_list.append(pf)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }
    if end_epoch == 30:
        print('\n******* Epoch [%3d/%3d], Training: AUC %.5f  PF %.5f *******\n' % (epoch+1, end_epoch, auc, pf))
        if auc >= max(train_auc_list) :
            torch.save(state, weights_folder + '/best.pt')
            print('saving the best model')
    else:
        print('\n******* Epoch [%3d/%3d], Validation AUC: %.5f PF %.5f *******\n' % (epoch+1, end_epoch+1, auc, pf))

train_auc_list = []
val_auc_list = []
train_pf_list = []
for i in range(epoch):
    train(model, optimizer, criterion, train_loader, device, i, 29)
    #scheduler.step()
    val(model, train_loader, device, train_auc_list, weights, i, 30 )
    val(model, val_loader, device, val_auc_list,weights, i, 29 )

model.load_state_dict(torch.load(weights + '/best.pt')['net'])
model = model.to(device)
print('start testing the model')
val(model, val_loader, device, val_auc_list, weights,i, 29 )
