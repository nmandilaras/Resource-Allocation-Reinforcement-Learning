import numpy as np
import random
import pickle
import constants
import math


class Agent:
    def __init__(self, env, dimensions, lr=0.1, gamma=1, epsilon=1):
        """

        :param env:
        :param lr:
        :param gamma:
        """
        self.env = env
        # q_table has the dimensions of each variable (state) and an extra one for every possible action from each state
        self.q_table = np.zeros(dimensions + [env.action_space.n])
        self.lr = lr  # usually
        self.gamma = gamma  # usually 0.8 - 0.99
        self.epsilon = epsilon

    def choose_action(self, cur_state, train=True):
        """"""
        if random.uniform(0, 1) < self.epsilon and train:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[cur_state])

    def adjust_exploration(self, i_episode):
        self.epsilon = max(0.1, min(1.0, 1.0 - math.log10((i_episode + 1) / np.prod(self.q_table.shape[:-1]))))  # /= np.sqrt(i_episode + 1)

    def adjust_lr(self, i_episode):
        self.lr = max(0.1, min(1.0, 1.0 - math.log10((i_episode + 1) / np.prod(self.q_table.shape[:-1]))))

    # def save_checkpoint(self, filename):
    #     with open(filename, 'wb') as f:
    #         pickle.dump([X_train, y_train], f)
    #
    # def load_checkpoint(self, filename):
    #     with open(filename, 'rb') as f:
    #         var_you_want_to_load_into = pickle.load(f)

    def update(self, *args):
        raise NotImplementedError

    # def train(self):
    #     __train_loop()
    #
    # def evaluate(self):
    #     __train_loop
