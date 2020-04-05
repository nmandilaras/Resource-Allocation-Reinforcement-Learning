import math
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, num_of_actions, gamma=0.999, epsilon=1):
        """
        """
        self.num_of_actions = num_of_actions  # TODO remove this it can be deduced.
        self.gamma = gamma  # usually 0.8 - 0.99
        self.epsilon = epsilon
        self.ada_divisor = 25  # np.prod(self.q_table.shape[:-1])

    def adjust_exploration(self, i_episode):
        self.epsilon = max(0.01, min(0.3, 1.0 - math.log10((i_episode + 1) / self.ada_divisor)))
        #  /= np.sqrt(i_episode + 1)

    @abstractmethod
    def choose_action(self, cur_state, train=True):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, filename):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, filename):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError
