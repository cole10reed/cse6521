import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.act1 = nn.ReLU()
        # self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten(start_dim=0)
 
        self.fc3 = nn.Linear(32768, 512)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, 1)
        nn.init.xavier_uniform_(self.fc4.weight)
        # self.act4 = nn.LeakyReLU()
 
    def forward(self, x):
        # input 3x64x64, output 32x64x64
        x = self.act1(self.conv1(x))
        # x = self.drop1(x)
        # input 32x64x64, output 32x64x64
        x = self.act2(self.conv2(x))
        # input 32x64x64, output 32x32x32
        x = self.pool2(x)
        # input 32x32x32, output 32768
        x = self.flat(x)
        # input 32768, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 1
        x = self.fc4(x)
        # x = self.act4(self.fc4(x))
        return x