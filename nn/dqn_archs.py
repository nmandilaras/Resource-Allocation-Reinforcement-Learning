import torch
import torch.nn as nn


class ClassicDQN(nn.Module):

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        self.output = nn.Linear(output_dim, actions_dim)

    def forward(self, x):
        return self.output(x)


class Dueling(nn.Module):

    def __init__(self, output_dim, actions_dim):
        super().__init__()
        # TODO consider refactoring the network arch, streams can consist of more layers
        self.value_stream = nn.Sequential(
            # nn.Linear(output_dim, 12),
            # nn.Dropout(p=0.1),
            # nn.ReLU(),
            # nn.Linear(12, 1)
            nn.Linear(output_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            # nn.Linear(output_dim, 12),
            # nn.Dropout(p=0.1),
            # nn.ReLU(),
            # nn.Linear(12, actions_dim)
            nn.Linear(output_dim, actions_dim)
        )

    def forward(self, x):
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        q_values = values + (advantages - advantages.mean())  # broadcasting happens on values

        return q_values
