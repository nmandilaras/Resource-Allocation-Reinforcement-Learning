import torch.nn as nn
from nn.dqn_archs import ClassicDQN


class ActorCritic(nn.Module):

    def __init__(self, features_dim, layers_dim, actions_dim, dropout=0.1):
        super().__init__()
        layers_in = [features_dim] + layers_dim[:-1]
        layers_out = layers_dim
        output_dim = layers_dim[-1]
        self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(in_feats, out_feats),
                                                    nn.Dropout(p=dropout),
                                                    nn.ELU())
                                      for in_feats, out_feats in zip(layers_in, layers_out)])

        self.action_head = nn.Linear(output_dim, actions_dim)

        self.value_head = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.layers(x)

        action_logits = self.action_head(x)

        state_value = self.value_head(x)

        return action_logits, state_value
