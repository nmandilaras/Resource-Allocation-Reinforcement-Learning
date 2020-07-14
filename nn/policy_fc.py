import torch.nn as nn
from nn.dqn_archs import ClassicDQN


class PolicyFC(nn.Module):

    def __init__(self, features_dim, layers_dim, actions_dim, dqn_arch=ClassicDQN, dropout=0.1):
        super().__init__()
        layers_in = [features_dim] + layers_dim[:-1]
        layers_out = layers_dim
        output_dim = layers_dim[-1]
        self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(in_feats, out_feats),
                                                    # nn.Dropout(p=dropout),  # weird connection in graphs
                                                    nn.ELU())
                                      for in_feats, out_feats in zip(layers_in, layers_out)])
        self.output = dqn_arch(output_dim, actions_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)

        return x
