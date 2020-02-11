import numpy as np
import random


class Agent:
    def __init__(self, env, var_freq, lr=0.01, gamma=0.9, epsilon=1):
        """

        :param env:
        :param lr:
        :param gamma:
        """
        self.env = env
        self.q_table = np.zeros(var_freq + [env.action_space.n])
        self.lr = lr
        self.gamma = gamma  # usually 0.8 - 0.99
        self.epsilon = epsilon

    def choose_action(self, cur_state):
        return self.env.action_space.sample() if random.uniform(0, 1) < self.epsilon else np.argmax(self.q_table[cur_state])

    def update(self, cur_state, action, new_state, reward):
        self.q_table[cur_state + (action,)] = (1 - self.lr) * self.q_table[cur_state + (action,)] + \
                                          self.lr * (reward + self.gamma * np.max(self.q_table[new_state, :]))

    def train(self):
        pass

    def evaluate(self):
        pass
