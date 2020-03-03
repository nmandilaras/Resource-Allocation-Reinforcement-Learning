import numpy as np
import random
import math
from agents.agent import Agent


class MCAgent(Agent):
    def __init__(self, num_of_actions, dimensions, gamma=0.999, epsilon=1):
        """
        """
        super().__init__(num_of_actions, gamma, epsilon)
        self.num_of_actions = num_of_actions
        # q_table has the dimensions of each variable (state) and an extra one for every possible action from each state
        self.q_table = np.zeros(dimensions + [num_of_actions])

    def choose_action(self, cur_state, train=True):
        """"""
        if random.uniform(0, 1) < self.epsilon and train:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_table[cur_state])

    def update(self, summary):
        visited_states = set()
        for state, action, value in summary:
            if (state, action) not in visited_states:
                
                visited_states.add((state, action))
        pass