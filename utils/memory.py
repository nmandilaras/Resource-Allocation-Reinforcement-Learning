import random
from collections import namedtuple
# from torch.utils.data import Dataset

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Memory:  # (Dataset):

    def __init__(self, capacity):  # TODO implement as a queue?
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # returns a list of samples(tuples)

    def flush(self):  # if we use flush in every episode it doesn't get trained at all
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]