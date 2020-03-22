import numpy as np
import random
import math
from agents.agent import Agent


class MCAgent(Agent):
    """ Monte carlo without exploration start. This means that we don't have the option to start the simulation
     from a different point every time but instead we explore the action space by using e-greedy"""

    def __init__(self, num_of_actions, dimensions, gamma=0.999, epsilon=1):
        """
        """
        super().__init__(num_of_actions, gamma, epsilon)
        self.num_of_actions = num_of_actions
        # q_table has the dimensions of each variable (state) and an extra one for every possible action from each state
        self.q_table = np.zeros(dimensions + [num_of_actions])
        self.num_of_visits = np.zeros(dimensions + [num_of_actions])  # we keep the number of times that we have visited
        # each Q(s, a)

    def choose_action(self, cur_state, train=True):
        """"""
        if random.uniform(0, 1) < self.epsilon and train:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_table[cur_state])

    def update(self, summary):
        """ Method that runs at the end of every episode, we take into consideration only the first time that we
        encounter a state. For each (state, action) we have the corresponding value. We update the mean Q(s, a) by
        taking into consideration the value of the current episode."""

        visited_states = set()
        for state, action, value in summary:
            if (tuple(state), action) not in visited_states:
                self.q_table[state + (action,)] = (self.q_table[state + (action,)] * self.num_of_visits[
                    state + (action,)] + value) / (self.num_of_visits[state + (action,)] + 1)
                self.num_of_visits[state + (action,)] += 1
                visited_states.add((tuple(state), action))

    def load_checkpoint(self, filename):
        pass

    def save_checkpoint(self, filename):
        pass
