from abc import ABC, abstractmethod

import math
import random
import torch
from utils.constants import *
from utils.memory import Memory, Transition
from agents.agent import Agent


class DeepAgent(Agent, ABC):

    def __init__(self, num_of_actions, network, criterion, optimizer, mem_size=1000, batch_size=32, gamma=0.999,
                 epsilon=1):
        """

        """
        super().__init__(num_of_actions, gamma, epsilon)
        self.device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device) seems slower with gpu
        self.memory = Memory(mem_size)
        self.policy_net = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    def choose_action(self, state, train=True):
        state = torch.tensor(state, device=self.device)

        if random.random() < self.epsilon and train:
            return torch.tensor([[random.randrange(self.num_of_actions)]], dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # we change max(1) to max(0) since we have only one element in this forward pass
                return self.policy_net(state).max(0)[1].view(1, 1)

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass

    @abstractmethod
    def compute_loss(self, state, action, reward, non_final_next_state, non_final_mask):
        raise NotImplementedError

    def update(self):
        try:
            transitions = self.memory.sample(self.batch_size)
        except ValueError:
            # print("Memory smaller than batch size")
            return

        # creates a tuple that contains all states as state
        sample_batch = Transition(*zip(*transitions))
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).

        state = torch.tensor(sample_batch.state, device=self.device)  # Concatenates the given sequence of seq tensors
        action = torch.tensor(sample_batch.action, device=self.device)
        reward = torch.tensor(sample_batch.reward, device=self.device)
        non_final_next_state = torch.tensor([s for s in sample_batch.next_state if s is not None], device=self.device)

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                sample_batch.next_state)), dtype=torch.bool, device=self.device)

        state_action_values, expected_state_action_values = self.compute_loss(state, action, reward, non_final_next_state, non_final_mask)
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()  # computes gradients
        # for param in policy_net.parameters(): # what is needed for ?
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # updates weights

    def push_in_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def adjust_exploration(self, steps_done):
        self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    def update_target_net(self):
        raise NotImplementedError
