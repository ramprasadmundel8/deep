import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
def generate_data():
    t = np.linspace(0, 20, 100)
    y = np.sin(t) + np.random.normal(scale=0.5, size=t.shape)
    return y

data = generate_data()
plt.plot(data)
plt.title('Synthetic Time Series Data')
plt.show()

# Prepare the dataset
def create_inout_sequences(input_data, tw): # tw-> time step window
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 10 # Number of time steps to look back
data = torch.FloatTensor(data).view(-1)
sequences = create_inout_sequences(data, seq_length)

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        rnn_out, hidden = self.rnn(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]

model = RNN()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for i in range(epochs):
    for seq, labels in sequences:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f'Epoch {i} loss: {single_loss.item()}')

with torch.no_grad():
    preds = []
    for seq, _ in sequences:
        preds.append(model(seq).item())

plt.plot(data.numpy(), label='Original Data')
plt.plot(np.arange(seq_length, seq_length + len(preds)), preds, label='Predicted')
plt.legend()
plt.show()
