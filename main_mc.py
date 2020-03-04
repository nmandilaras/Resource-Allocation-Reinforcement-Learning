import gym
import numpy as np
from utils import constants, algorithms
from itertools import count
from agents.mc_agent import MCAgent
import logging.config
from utils.quantization import Quantization

env = gym.make(constants.environment)
num_episodes = 300
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
    state = env.reset()
    agent.adjust_exploration(i_episode)

    for t in count():
        # env.render()
        # Select and perform an action
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action.item())

        if done:
            next_state = None

        state = next_state

        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    agent.update()
