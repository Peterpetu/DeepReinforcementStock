import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Adjust the input size of the first linear layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])  # state_size should be 602


        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        x = state.view(state.size(0), -1)  # Flatten the state representation
        for each in self.hidden_layers:
            x = F.relu(each(x))
        return self.output(x)
