import unittest
import gym
from utils import constants
import numpy as np
import torch
from nn.policy_fc import PolicyFC
from utils.memory import Memory, Transition
import torch.optim as optim
from agents.dqn_agents import DQNAgent, DDQNAgent

BATCH_SIZE = 16


class TestRL(unittest.TestCase):
    lr = 1e-2
    layers_dim = [6, 6]
    mem_size = 100

    env = gym.make(constants.environment)

    num_of_observations = env.observation_space.shape[0]
    num_of_actions = env.action_space.n

    memory = Memory(mem_size)
    network = PolicyFC(env.observation_space.shape[0], [24, 12], env.action_space.n)
    criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(network.parameters(), lr)
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, mem_size)

    def test_forward(self):

        state = np.float32(self.env.reset())
        state = torch.tensor(state)  # .double()  # network is in float so input must be also

        output = self.agent.policy_net(state)

        print(output)
        print(output.max(0)[1].item())  # result is good the index of max action is selected

    def test_forward_batch(self):
        for _ in range(100):
            state = self.env.reset()
            state = np.float32(state)
            # state is a np array
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(state, action, next_state, reward)

        # transitions = DataLoader(self.memory, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        #
        # # issue we don't need to format all the batches but just one, maybe dataloader is not the way to go.
        # for i, x in enumerate(transitions):
        #     output = self.policy_net(x.state)
        #     self.assertEqual([len(x.state), self.env.action_space.n], list(output.shape))
        #     print(output)
        #     result = output.max(1)
        #     print(result)
        #     result = result[1]
        #     print(result)
        #     print(result.view(1, 1))
        #     break

        transitions = self.memory.sample(BATCH_SIZE)

        # print(type(transitions))
        # print(len(transitions))
        # print(transitions[0])

        batch = Transition(*zip(*transitions))

        # *transitions unpacks the list of transitions so we have 16 transitions each with 4 elements
        # it would be the same if we had 16 lists each with 4 elements instead of 16 tuples
        # zip creates a list of 4 tuples each with 16 elements

        # for x in zip(*transitions):
        #     print(type(x[0]))
        #     ts = torch.tensor(x)
        #     print(ts)
        #     print(ts.shape)
        #     print(x[0])
        #     break

        batch_state = torch.tensor(batch.state)
        #
        #
        # print(batch_state)
        output = self.network(batch_state)
        self.assertEqual([len(batch_state), self.env.action_space.n], list(output.shape))
        print(output)
        result = output.max(1)
        print(result)
        result = result[1]
        print(result)
        print(result.view(1, 1))

