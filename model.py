import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[64, 32], activation=nn.ReLU):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        input_size = state_size

        for hidden_size in hidden_layers:
            try:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(activation())
                input_size = hidden_size
            except Exception as e:
                print(f"Error occurred while creating hidden layer: {e}")

        try:
            self.layers.append(nn.Linear(input_size, action_size))
        except Exception as e:
            print(f"Error occurred while creating output layer: {e}")

    def forward(self, x):
        try:
            for layer in self.layers:
                x = layer(x)
            return x
        except Exception as e:
            print(f"Error occurred during forward pass: {e}")
            return None
