import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[64, 32], activation=nn.ReLU):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        input_size = state_size

        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(activation())
            input_size = hidden_size

        self.layers.append(nn.Linear(input_size, action_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
