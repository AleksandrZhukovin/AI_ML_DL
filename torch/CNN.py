import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

# Set GPU device
cuda0 = torch.device('cuda:0')


# Model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2).cuda(0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2).cuda(0)
        self.pl1 = nn.Linear(7*7*64, 256).cuda(0)
        self.pl2 = nn.Linear(256, 10).cuda(0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(-1, 7*7*64)
        x = self.pl1(x)
        x = F.relu(x)

        x = self.pl2(x)
        return x


# Load data
train = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, shuffle=False)


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train
with torch.cuda.device(0):
    for epoch in trange(3):
        for images, labels in tqdm(train_loader):

            optimizer.zero_grad()

            x = images.cuda(0)
            y = model(x)
            loss = criterion(y, labels.cuda(0))
            loss.backward()
            optimizer.step()


# Test
correct = 0
total = len(test)

with torch.cuda.device(0):
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            x = images.cuda(0)
            y = model(x)

            prediction = torch.argmax(y, dim=1)
            correct += torch.sum((prediction == labels.cuda(0)).float())


print(f'{correct} of {total}')
