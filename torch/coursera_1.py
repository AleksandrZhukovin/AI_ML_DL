import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Load data
train = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, shuffle=False)

# Init weights
W = torch.rand(784, 10)/np.sqrt(784)
W.requires_grad_()
b = torch.zeros(10, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W, b], lr=0.1)

# Train
for images, labels in tqdm(train_loader):
    optimizer.zero_grad()

    x = images.view(-1, 28*28)
    y = torch.matmul(x, W) + b
    cross_entropy = F.cross_entropy(y, labels)
    cross_entropy.backward()
    optimizer.step()

# Test
correct = 0
total = len(test)

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        x = images.view(-1, 28*28)
        y = torch.matmul(x, W) + b

        prediction = torch.argmax(y, dim=1)
        correct += torch.sum((prediction == labels).float())

# Feed random data
with open('mnist_test.csv', 'r') as check_file:
    data = []
    l = []
    for i in check_file.readlines():
        data.append(i.split(',')[1:])
        l.append(i.split(',')[0])
    x = np.reshape(data[100], (1, 784))
    x = x.astype(np.int)
    x = torch.from_numpy(x).float()
    y = torch.matmul(x, W) + b
    print(torch.argmax(y, dim=1), l[100])

print(f'{correct} of {total}')
