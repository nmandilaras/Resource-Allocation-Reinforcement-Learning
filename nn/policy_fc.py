import torch.nn as nn


class PolicyFC(nn.Module):

    def __init__(self, observations_dim, hidden_dim, output_dim, actions_dim, dqn_arch):
        super().__init__()
        # TODO consider refactoring the network arch, make variant the number of connected layers
        self.input = nn.Sequential(nn.Linear(observations_dim, output_dim),
                     nn.Dropout(p=0.1),
                    nn.ReLU())
        # self.hidden = nn.Linear(hidden_dim, output_dim)
        self.output = dqn_arch(output_dim, actions_dim)

    def forward(self, x):
        x = self.input(x)
        # x = F.relu(self.hidden(x))
        x = self.output(x)

        return x
