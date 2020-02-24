import torch
import torch.optim as optim
import gym
from utils import constants, algorithms
import matplotlib.pyplot as plt
import numpy as np
from nn.dqn import DQN
from nn.dueling import Dueling
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dqn_agent import DQNAgent
from itertools import count

BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10  # target net is updated with the weights of policy net once every 10 episodes
num_episodes = 30

env = gym.make(constants.environment)

arch = algorithms.DUELING
# choose architecture
if arch == algorithms.DQN:
    network = DQN(env.observation_space.shape[0], 24, 12, env.action_space.n)
elif arch == algorithms.DUELING:
    network = Dueling(env.observation_space.shape[0], 24, 12, env.action_space.n)
else:
    raise NotImplementedError

criterion = torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.RMSprop(network.parameters())

double = False
# choose agent
if double:
    agent = DoubleDQNAgent(env.action_space.n, network, criterion, optimizer)
else:
    agent = DQNAgent(env.action_space.n, network, criterion, optimizer)

steps_done = 0
episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = np.float32(state)  # torch.from_numpy(state).float()
    for t in count():
        # env.render()
        # Select and perform an action
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = np.float32(next_state)
        reward = torch.tensor([reward])

        if done:
            next_state = None

        agent.push_in_memory(state, action, next_state, reward)  # Store the transition in memory
        state = next_state

        agent.update()  # Perform one step of the optimization (on the policy network)
        agent.adjust_exploration(i_episode)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    if i_episode % TARGET_UPDATE == 0:  # Update the target network
        agent.update_target_net()

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
