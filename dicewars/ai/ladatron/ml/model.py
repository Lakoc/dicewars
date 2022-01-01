import torch
from torch import nn


class Network(torch.nn.Module):

    def __init__(self, input_features, output_features):
        super(Network, self).__init__()
        hidden_dims = [32, 16]
        self.linear1 = nn.Linear(input_features, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], output_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        return y
