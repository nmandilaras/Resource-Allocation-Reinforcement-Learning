from abc import ABC

import numpy as np
import random
import math
from agents.agent import Agent


class TDAgent(Agent, ABC):
    def __init__(self, num_of_actions, dimensions, lr=0.1, gamma=0.999, epsilon=1):
        """
        """
        super().__init__(num_of_actions, gamma, epsilon)
        self.num_of_actions = num_of_actions
        # q_table has the dimensions of each variable (state) and an extra one for every possible action from each state
        self.q_table = np.zeros(dimensions + [num_of_actions])
        self.lr = lr

    def choose_action(self, cur_state, train=True):
        """"""
        if random.uniform(0, 1) < self.epsilon and train:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_table[cur_state])

    def adjust_lr(self, i_episode):
        self.lr = max(0.01, min(1.0, 1.0 - math.log10((i_episode + 1) / self.ada_divisor)))
        #  (sum(self.q_table.shape)*2)

    def save_checkpoint(self, filename):
        pass
    #     with open(filename, 'wb') as f:
    #         pickle.dump([X_train, y_train], f)

    def load_checkpoint(self, filename):
        pass
    #     with open(filename, 'rb') as f:
    #         var_you_want_to_load_into = pickle.load(f)
