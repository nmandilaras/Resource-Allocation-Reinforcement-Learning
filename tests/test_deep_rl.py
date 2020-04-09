import unittest
import gym
from utils import constants
import numpy as np
import torch
from nn.policy_fc import PolicyFC
from utils.memory import Memory
import torch.optim as optim
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from nn.dqn_archs import ClassicDQN, Dueling

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
    network.eval()
    criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(network.parameters(), lr)
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, mem_size)

    def test_Memory(self):
        self.memory.push(1, 2, 3, 4, 5)
        state, action, next_state, reward, done = self.memory.sample(1)[0]
        print(state, action, next_state, reward, done)
        print(str(ClassicDQN))

    def test_forward(self):

        state = np.float32(self.env.reset())   # network is in float so input must be also
        state = torch.tensor(state)

        output = self.agent.policy_net(state)

        print(output)
        print(output.max(0)[1].item())  # result is good the index of max action is selected

    def test_forward_batch(self):
        for _ in range(100):  # it will put into memory only initial states
            state = self.env.reset()
            state = np.float32(state)
            # state is a np array
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.float32(next_state)
            self.memory.push(state, action, next_state, reward, done)

        transitions = self.memory.sample(BATCH_SIZE)

        # print(type(transitions))
        # print(len(transitions))
        # print(transitions[0])

        batch = zip(*transitions)

        state, action, next_state, reward, done = list(map(lambda x: torch.tensor(x), batch))
        # for b in [state, action, next_state, reward, done]:
        #     print(b.type())
        #     print(b.shape)

        #
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

        predicted_qs = self.network(state)
        print(predicted_qs)
        print(action)
        result = predicted_qs.gather(1, action)
        print(result)
        print(result.shape)

        # policy_actions = self.network(next_state)
        # print(policy_actions)
        # # print(policy_actions.max(1)[0])
        # print(policy_actions.max(1)[1])
        # next_state_values = self.network(next_state).gather(1, policy_actions.max(1)[1].unsqueeze(1))
        # print('here')
        # print(next_state_values.squeeze(1))

        # self.assertEqual([len(state), self.env.action_space.n], list(output.shape))  # batch_size x 2
        # print(output)
        # next_state_values = self.network(next_state).max(1)[0].detach()
        # print(next_state_values)
        # # max of each row, then we keep the value, not the index
        # expected_q_values = reward + (1 - done.int()) * 0.999 * next_state_values
        # print(done.int())
        # print(reward[:4] + (1 - torch.tensor([0, 0, 1, 0], dtype=torch.int32)))
        # print(expected_q_values.shape)

        # print(result)
        # result = result[1]
        # print(result)
        # print(result.view(1, 1))

