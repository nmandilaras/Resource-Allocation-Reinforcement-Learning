import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, observations_dim, hidden_dim, output_dim, actions_dim):
        super().__init__()
        self.input = nn.Linear(observations_dim, output_dim)
        # self.hidden = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, actions_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        # x = F.relu(self.hidden(x))
        x = self.output(x)

        return x
