import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

data = torch.randint(0, 1000, (100, 10))
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, vocabsize, embeddingdim, hiddendim, outputdim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocabsize, embeddingdim)
        self.lstm = nn.LSTM(embeddingdim, hiddendim, batch_first=True)
        self.fc = nn.Linear(hiddendim, outputdim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))

model = LSTMClassifier(1000, 50, 100, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def train(n_epochs):
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for data, tgts in loader:
            outputs = model(data)
            loss = criterion(outputs, tgts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

train(10)
