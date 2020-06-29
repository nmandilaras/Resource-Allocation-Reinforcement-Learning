import random
from collections import deque


class Memory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, *args):
        """Saves a transition."""
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None  # returns a list of samples(tuples)

    def batch_update(self, tree_idx, abs_errors):
        pass

    # def flush(self):  # if we use flush in every episode it doesn't get trained at all
    #     self.memory.clear()
