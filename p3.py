import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Linear(64 * 8 * 8, 512)
        self.dense2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

class CNNWithBNDropout(nn.Module):
    def __init__(self):
        super(CNNWithBNDropout, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Linear(64 * 8 * 8, 512)
        self.dense2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(train_dataset, range(200))
test_subset = Subset(test_dataset, range(50))

train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

# Function to train and evaluate a model
def train(model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = torch.argmax(output.data, dim=1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                predicted = torch.argmax(output.data, dim=1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct_test / total_test
        
        print(f'Epoch: [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {train_acc:.4f}%, Test Loss: {avg_test_loss:.4f}, '
              f'Test Accuracy: {test_acc:.4f}%')

model1 = SimpleCNN()
model2 = CNNWithBNDropout()
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
train(model1, optimizer1, criterion, 20)
train(model2, optimizer2, criterion, 30)
