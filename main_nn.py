import torch
import torch.optim as optim
import gym
from utils import constants, algorithms
import matplotlib.pyplot as plt
import numpy as np
from nn.dqn import DQN
from nn.dueling import Dueling
from agents.double_dqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from itertools import count
from utils.functions import plot_durations

TARGET_UPDATE = 10  # target net is updated with the weights of policy net once every 10 episodes
EVAL_INTERVAL = 10
num_episodes = 1000

env = gym.make(constants.environment)

arch = algorithms.DUELING  # dueling can use DQN or DDQN as well
# choose architecture
if arch == algorithms.DQN:
    network = DQN(env.observation_space.shape[0], 24, 12, env.action_space.n)
elif arch == algorithms.DUELING:
    network = Dueling(env.observation_space.shape[0], 24, 12, env.action_space.n)
else:
    raise NotImplementedError

criterion = torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters())

double = True
# choose agent
if double:
    agent = DDQNAgent(env.action_space.n, network, criterion, optimizer)
else:
    agent = DQNAgent(env.action_space.n, network, criterion, optimizer)

steps_done = 0
episode_durations = {}
eval_durations = {}
means = []


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = np.float32(state)  # torch.from_numpy(state).float()
    train = True
    if (i_episode + 1) % EVAL_INTERVAL == 0:
        train = False
    for t in count():
        steps_done += 1
        # env.render()
        # Select and perform an action
        action = agent.choose_action(state, train=train)
        next_state, reward, done, _ = env.step(action.item())
        next_state = np.float32(next_state)
        reward = torch.tensor([reward])

        if done:
            next_state = None

        agent.push_in_memory(state, action, next_state, reward)  # Store the transition in memory
        state = next_state

        agent.update()  # Perform one step of the optimization (on the policy network)
        agent.adjust_exploration(steps_done)
        if done:
            if train:
                episode_durations[i_episode] = (t + 1)
            else:
                eval_durations[i_episode] = (t + 1)

            plot_durations(episode_durations, means, eval_durations)
            break

    if i_episode % TARGET_UPDATE == 0:  # Update the target network
        agent.update_target_net()

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
