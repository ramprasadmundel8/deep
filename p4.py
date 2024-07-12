import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_subset = Subset(train_dataset, range(200))
test_subset = Subset(test_dataset, range(50))

train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def sgd_update(parameters, lr):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                param.data -= lr * param.grad.data
                param.grad.zero_()

class CustomAdagrad(Optimizer):
    def __init__(self, parameters, lr=0.01, epsilon=1e-10):
        self.parameters = list(parameters)
        self.lr = lr
        self.epsilon = epsilon
        self.sum_squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for param, sum_sq_grad in zip(self.parameters, self.sum_squared_gradients):
                if param.grad is not None:
                    sum_sq_grad += param.grad.data ** 2
                    adjusted_lr = self.lr / (self.epsilon + torch.sqrt(sum_sq_grad))
                    param.data -= adjusted_lr * param.grad.data
                    param.grad.zero_()

    def zero_grad(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param.grad.zero_()

device = torch.device('cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

def train_model(num_epochs, optimizer_choice='adagrad'):
    if optimizer_choice == 'sgd':
        optimizer = None
    else:
        optimizer = CustomAdagrad(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if optimizer_choice == 'sgd':
                sgd_update(model.parameters(), lr=0.01)
            else:
                optimizer.step()
            train_loss += loss.item()
            predicted = torch.argmax(output.data, dim=1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        model.eval()
        test_loss = 0
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
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.8f}%, '
              f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.8f}%')

train_model(5, optimizer_choice='adagrad')
