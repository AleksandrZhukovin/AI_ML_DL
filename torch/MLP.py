import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 500)
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        h = self.l1(x)
        h = F.relu(h)
        return self.l2(h)


# Load data
train = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, shuffle=False)

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train
for images, labels in train_loader:

    optimizer.zero_grad()

    x = images.view(-1, 28*28)
    out = model.forward(x)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

correct = 0
total = len(test)

with torch.no_grad():
    for images, labels in test_loader:
        x = images.view(-1, 28*28)
        y = model.forward(x)

        prediction = torch.argmax(y, dim=1)
        correct += torch.sum((prediction == labels).float())

print(f'{correct} of {total}')
