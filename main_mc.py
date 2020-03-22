import gym
import numpy as np
from utils import constants, algorithms
from itertools import count
from agents.mc_agent import MCAgent
import logging.config
from utils.quantization import Quantization
from utils.functions import plot_durations
import matplotlib.pyplot as plt

EVAL_INTERVAL = 10

env = gym.make(constants.environment)
num_episodes = 300
train_durations = {}
eval_durations = {}
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

num_of_actions = env.action_space.n
high_intervals = env.observation_space.high
low_intervals = env.observation_space.low

logger.debug(high_intervals)
logger.debug(low_intervals)

vars_ls = list(zip(low_intervals, high_intervals, constants.var_freq))
quantizator = Quantization(vars_ls, lambda x: [x[i] for i in [0, 1, 2, 3]])
agent = MCAgent(num_of_actions, quantizator.dimensions)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    train = True
    if (i_episode + 1) % EVAL_INTERVAL == 0:
        train = False

    observation = env.reset()
    state = quantizator.digitize(observation)
    agent.adjust_exploration(i_episode)
    sar_ls = []

    for t in count():
        # env.render()
        action = agent.choose_action(state, train=train)    # Select and perform an action
        next_observation, reward, done, _ = env.step(action)

        sar_ls.append((state, action, reward))

        if done:
            if train:
                train_durations[i_episode] = (t + 1)
            else:
                eval_durations[i_episode] = (t + 1)
            plot_durations(train_durations, None, eval_durations)
            break

        next_state = quantizator.digitize(next_observation)
        state = next_state

    summary = []
    value = 0

    for state, action, reward in reversed(sar_ls):
        value = reward + agent.gamma * value
        summary.append((state, action, value))

    agent.update(reversed(summary))

env.close()
plt.show()
