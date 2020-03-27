import torch
import torch.nn as nn

class PolicyConv(nn.Module):
    """ To be used if we choose to get the screen outputs as initial features """
    def __init__(self, observations_dim, hidden_dim, output_dim, actions_dim, dqn_arch):
        super().__init__()
        # TODO consider refactoring the network arch, make variant the number of connected layers
        # self.input =
        self.output = dqn_arch(output_dim, actions_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.output(x)

        return x