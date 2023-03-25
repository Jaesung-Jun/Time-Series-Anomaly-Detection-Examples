from unicodedata import bidirectional
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Encoder(nn.Module):
    def __init__(self, input, output):
        super(Encoder, self).__init__()
        #self.transformer = nn.Transformer(nhead = )
        self.bilstm = nn.LSTM(input_size=input, hidden_size = 100, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(20, output)

    def forward(self, x):
        x = self.bilstm(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x

class Generator(nn.Module):
    def __init__(self, input, output):
        super(Generator, self).__init__()
        self.bilstm = nn.LSTM(input, hidden_size=64, bidirectional=True)
        self.bilstm = nn.LSTM(input, hidden_size=64, bidirectional=True)
        self.fc_layer = nn.Linear(20, output)
        
    def forward(self, x):

encoder = Encoder(120, 20).to(device)
print(encoder)
