import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset = Subset(dataset, range(1000))
dataloader = DataLoader(subset, batch_size=10, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optim_gen = optim.Adam(generator.parameters(), lr=2e-4)
optim_disc = optim.Adam(discriminator.parameters(), lr=2e-4)

def train(num_epochs):
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        for real, _ in dataloader:
            real = real.view(-1, 28*28)
            batch_size = real.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, 100)
            fake = generator(noise)
            disc_real = discriminator(real)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # Backprop
            optim_disc.zero_grad()
            loss_disc.backward()
            optim_disc.step()

            # Train Generator
            noise = torch.randn(batch_size, 100)
            fake = generator(noise)
            disc_fake = discriminator(fake)
            loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))

            # Backprop
            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()

        print(f'Epoch {epoch+1}, Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f}')

train(15)
