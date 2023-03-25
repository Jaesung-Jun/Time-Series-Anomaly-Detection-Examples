import torch
from torch import nn

class Simple_LSTM(nn.Module):
    def __init__(self, input, output):
        super(Simple_LSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input, hidden_size = 100, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(2, output)

    def forward(self, input, output):
        x = 