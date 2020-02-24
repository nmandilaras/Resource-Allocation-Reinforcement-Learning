import torch
import torch.nn as nn
import torch.nn.functional as F


class Dueling(nn.Module):

    def __init__(self, observations_dim, hidden_dim, output_dim, actions_dim):
        super().__init__()
        self.input = nn.Linear(observations_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, output_dim)

        self.value_stream = nn.Linear(output_dim, 1)
        self.advantage_stream = nn.Linear(output_dim, actions_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        features = F.relu(self.hidden(x))

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = values + (advantages - advantages.mean())

        return q_values
