import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split # i dont feel like manually coding this
from sklearn.preprocessing import LabelEncoder  #i dont feel like manually coding this
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()
api.dataset_download_files('uciml/iris', unzip=True) #iris dataset

class NeuralNetwork(nn.Module): #nn.module is a parent class for ts
    def __init__(self, inputFeatures = 4, h1 = 8, h2 = 8, outputFeatures = 3):
        super().__init__() #idk what this does but you have to use it apprently
        #interconnect the layers
        self.fc1 = nn.Linear(inputFeatures, h1) 
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, outputFeatures)

    def forward(self, x): #using RELU for all layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
torch.manual_seed(67) # six sven :)

model = NeuralNetwork() #instance stuff

df = pd.read_csv('Iris.csv')
le = LabelEncoder()

x = df.drop(['Id', 'Species'], axis=1) #feature vectors
y = df['Species'] #labels
y = le.fit_transform(y) #string to int

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=67) # six seven

#tensor stuff
xTrainTensors = torch.FloatTensor(xTrain.values) #.values is to pass in a numpy array
yTrainTensors = torch.LongTensor(yTrain) #long tensor for int

xTestTensors = torch.FloatTensor(xTest.values) #.values is to pass in a numpy array
yTestTensors = torch.LongTensor(yTest) #long tensor for int

criterion = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #pass in the neural network

#Training
epochs = 100
losses = []

for i in range (epochs):
    yPred = model.forward(xTrainTensors)

    loss = criterion(yPred, yTrainTensors)

    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'Epoch: {i} and loss {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad(): #no back propogation
    yEval = model.forward(xTestTensors)
    loss = criterion(yEval, yTestTensors)

correct = 0
with torch.no_grad():
    for i, data in enumerate(xTestTensors):
        yVal = model.forward(data)

        print(f'{i+1}.) {str(yVal)} \t {yTestTensors[i]} \t {yVal.argmax().item()}')

        if yVal.argmax().item() == yTestTensors[i]:
            correct +=1

print(f'We got {correct} correct')

newVal = torch.tensor([6.7, 4.1, 4.2, 0.3]) #new point

with torch.no_grad():
    z = model.forward(newVal)
    print(f'We guessed element: {z.argmax().item()}')

#save model:
torch.save(model.state_dict(), 'irisModel.pt')

#load new model with following code
#new_model = NeuralNetwork()
#new_model.load_state_dict(torch.load('irisModel.pt'))