## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batch_norm = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*26*26, 544)
        self.fc2 = nn.Linear(544,272)
        self.fc3 = nn.Linear(272, 136)
        self.dropout = nn.Dropout(p=0.25)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = F.relu(self.conv1(x)) #32*222*222
        x = F.max_pool2d(x,2,2) #32*111*111
        x = self.dropout(F.relu(self.conv2(x))) #64*109*109
        x = F.max_pool2d(x,2,2) #64*54*54
        x = self.batch_norm(F.relu(self.conv3(x)))#128*52*52
        # x = self.adapt(x)
        # x = x.view(-1, 128*10*10)
        x = F.max_pool2d(x,2,2) #128*26*26
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        
        return x
