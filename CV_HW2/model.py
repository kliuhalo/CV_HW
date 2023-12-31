import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
from cnn_utils import *
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=s, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    
    
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
    


class ResModel(nn.Module):
    def __init__(self, n_class):
        super(ResModel,self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3,2,padding=1),
        )
        self.stage2 = nn.Sequential(
        # 3
            ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
            IndentityBlock(256, 3, [64, 64, 256]),
            IndentityBlock(256, 3, [64, 64, 256]),
        )
        self.stage3 = nn.Sequential(
        # 4
            ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
            IndentityBlock(512, 3, [128, 128, 512]),
        )
        self.stage4 = nn.Sequential(
        # 6
            ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
            IndentityBlock(1024, 3, [256, 256, 1024]),
        )
        self.stage5 = nn.Sequential(
        # 3
            ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
            IndentityBlock(2048, 3, [512, 512, 2048]),
            IndentityBlock(2048, 3, [512, 512, 2048]),
        )
        self.pool = nn.AvgPool2d(2,2,padding=1)
        self.fc = nn.Sequential(
            nn.Linear(32768, 1, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
