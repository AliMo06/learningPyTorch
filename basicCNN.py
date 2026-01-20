import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #fully connected layers
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10) #10 at the end since we have that many clases

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x,2,2)

        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x,2,2)

        x = x.view(-1, 16*5*5) #-1 so that we can vary batch size

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return f.log_softmax(x, dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(67)

#transform is a function that does images -> tensor
transform = transforms.ToTensor()
trainData = datasets.MNIST(root="/cnndata", train = True, download = True, transform = transform)
testData = datasets.MNIST(root="/cnndata", train = False, download = True, transform = transform)

#create batch 
trainLoader = DataLoader(trainData, batch_size=128, shuffle=True)
testLoader = DataLoader(testData, batch_size=128, shuffle=False)

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

startTime = time.time()

epochs = 5
trainLosses = []
testLosses = []
trainCorrects = []
testCorrects = []

#train
for i in range(epochs):
    trainCorrect = 0
    testCorrect = 0

    for b, (xTrain, yTrain) in enumerate(trainLoader):
        b+=1

        xTrain = xTrain.to(device) #gpu
        yTrain = yTrain.to(device)

        yPred = model(xTrain) #get predicted valuers from training set
        loss = criterion(yPred, yTrain)
        
        predicted = torch.max(yPred.data, 1)[1] #add up number of corrent predictions
        batchCorrect = (predicted == yTrain).sum() #how many corr from batch

        trainCorrect += batchCorrect
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 25 == 0:
            print(f"Epoch: {i} Batch: {b}  Loss:{loss.item()}")

    trainLosses.append(loss)
    trainCorrects.append(trainCorrect)

#test
with torch.no_grad(): #no gradient so that we dont update our eigts and biases with tests
    for b, (xTest, yTest) in enumerate(testLoader):
        xTest = xTest.to(device)
        yTest = yTest.to(device)

        yVal = model(xTest)
        predicted = torch.max(yVal.data, 1)[1]
        testCorrect += (predicted == yTest).sum()

    loss = criterion(yVal, yTest)
    testLosses.append(loss)
    testCorrects.append(testCorrect)
    test_acc = 100 * testCorrect / len(testData)
    print(f"Epoch {i} — Test Accuracy: {test_acc:.2f}%")

endTime = time.time()
totalTime = endTime - startTime

print(f"training took: {totalTime/60} minutes!")