import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

import os
import pandas as pd
from skimage import io
import signal

batch_size = 64
epochs = 5
learning_rate = 1e-3
weight_decay = 0.001
momentum = 0.9

with open('hyperparams.csv', newline='') as csvfile:

    params = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in params:
        match i:
            case 0:
                batch_size = int(row[1])
            case 1:
                epochs = int(row[1])
            case 2:
                learning_rate = float(row[1])
            case 3:
                weight_decay = float(row[1])
            case 4:
                momentum = float(row[1])
        i += 1

hiscore = 0

class SPARKDataset(DataLoader):
    def __init__(self, csv_file, root, transform=None):
        self.root_dir = root
        self.satInfo = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.satInfo)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.satInfo.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.satInfo.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

#Define training data.
training_data = SPARKDataset(
    csv_file="data/SPARK/newtrain.csv",
    root="data/SPARK/train+val",
    transform=ToTensor(),
)

#Define validation data.
val_data = SPARKDataset(
    csv_file="data/SPARK/newval.csv",
    root="data/SPARK/train+val",
    transform=ToTensor(),
)

#Define test data.
test_data = SPARKDataset(
    csv_file="data/SPARK/test_ground_truth.csv",
    root="data/SPARK/test",
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def imshow(img):
    img = img[:3, :, :]
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#define resnet residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Define model--we can also use models from torchvision.models
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*16, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):

        c1 = F.relu(self.conv1(input))

        s2 = F.max_pool2d(c1, (2, 2))

        c3 = F.relu(self.conv2(s2))

        print(c3.shape)
        s4 = F.max_pool2d(c3, 1)

        l5 = torch.flatten(s4, 1)
        print(l5.shape)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(l5))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)

        #fun = s4
        #fun = torchvision.utils.make_grid(fun.cpu())
        #fun = fun[:, :, :3]
        #print(fun.shape)
        #imshow(fun)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 11):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(147968, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model = ResNet(ResBlock, [3,4,6,3]).to(device)
print(model)

def handler(signum, frame):
    torch.save(model.state_dict(), "modelSave.pth")
    print("Saved PyTorch Model State to modelSave.pth")
    sys.exit(0)  

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, weight_decay, momentum)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        cor = 0
        #do image transforms here
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.25),
        ])

        X = transforms(X)

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        cor += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} % correct: {(cor/batch_size)*100} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


signal.signal(signal.SIGINT, handler)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")
print("High Score: ", str(hiscore))
test(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

classes = [
    "smart-1",
    "cheops",
    "lisa_pathfinder",
    "debris",
    "proba_3_ocs",
    "proba_3_csc",
    "soho",
    "earth_observation_sat_1",
    "proba_2",
    "xmm_newton",
    "double_star",
]
