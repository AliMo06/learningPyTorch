import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

tensor2d = torch.randn(2,3) #rows, columns
print("Device:", tensor2d.device)
print(tensor2d)

tensor3d = torch.randn(2,3,4) #depth/matrices, rows, columns
print(tensor3d)

tensor4d = torch.randn(5,5,5,5) #4d tensor that im moving to gpu
tensor4d = tensor4d.to(device) 
print(tensor4d.device) #says what its stored on, should be CUDA (it is)


#some cool ones that i wanted to try
arange = torch.arange(1, 12, 2) #start max, step, 1d tensor
print(arange)

arange = arange.reshape(2,3) #changes array shape
print(arange)

linspace = torch.linspace(0, 1, 8) # 8 evenly distributed elements from 0-1
print(linspace)

linspace = linspace.reshape(2,2,2)
print(linspace)

eye = torch.eye(3) #identity matrix/tensor
print(eye)

#play with activation functions
x = torch.tensor([-1.0, 0.0, 1.0])

print(torch.relu(x)) #max(0,x)
print(torch.sigmoid(x)) #1/1-e^-x
print(torch.tanh(x)) #tanh(x)
print(torch.softmax(x, dim=0)) #pobabilites sum to 1, converts all numbers to something between 0&1


#equivalence and stuff
x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 2, 2])

print(x == y)
print(x > y)
print(torch.all(x > 0))
print(torch.any(x == 2))

