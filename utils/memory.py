import random
from collections import deque


class Memory:

    def __init__(self, capacity):  # TODO implement as a queue?
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # returns a list of samples(tuples)

    def flush(self):  # if we use flush in every episode it doesn't get trained at all
        self.memory.clear()
