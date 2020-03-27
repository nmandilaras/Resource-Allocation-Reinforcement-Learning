import math
import random
import torch
from utils.constants import *
from agents.agent import Agent
from torch.distributions import Categorical


class Reinforce(Agent):

    def __init__(self, num_of_actions, network, criterion, optimizer, mem_size=1000, batch_size=32, gamma=0.999,
                 epsilon=1):
        """

        """
        super().__init__(num_of_actions, gamma, epsilon)
        self.device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device) seems slower with gpu
        self.policy_net = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

    def choose_action(self, state, train=True):
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            probs = self.policy_net(state)
            print(probs)
            m = Categorical(probs)
            print(m)
            action = m.sample()
            print(action.item())
            # policy.saved_log_probs.append(m.log_prob(action))
            return action.item()  # action is a tensor so we return just the number

    def update(self, *args):
        pass

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass