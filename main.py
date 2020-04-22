import gym
from utils import constants
from utils.constants import RLAlgorithms
import logging.config
from utils.quantization import Quantization
from agents.classic_agents.td_agents import QAgent, DoubleQAgent, SARSAgent
from utils.functions import plot_rewards
from itertools import count
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """The problem is considered solved when the poll stays upright for over 195 time steps, 100 times consecutively"""

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('simpleExample')

    env = gym.make(constants.environment)
    high_intervals = env.observation_space.high
    low_intervals = env.observation_space.low
    num_of_actions = env.action_space.n

    logger.debug(high_intervals)
    logger.debug(low_intervals)

    vars_ls = list(zip(low_intervals, high_intervals, constants.var_freq))
    quantizator = Quantization(vars_ls, lambda x: [x[i] for i in [0, 1, 2, 3]])

    logger.debug(quantizator.vars_bins)

    algorithm = RLAlgorithms.DOUBLE_Q_LEARNING

    if algorithm == RLAlgorithms.Q_LEARNING:
        agent = QAgent(num_of_actions, quantizator.dimensions)
    elif algorithm == RLAlgorithms.SARSA:
        agent = SARSAgent(num_of_actions, quantizator.dimensions)
    elif algorithm == RLAlgorithms.DOUBLE_Q_LEARNING:
        agent = DoubleQAgent(num_of_actions, quantizator.dimensions)
    else:
        raise NotImplementedError

    logger.debug(quantizator.dimensions)
    logger.debug(agent.q_table.shape)

    train_durations = {}
    eval_durations = {}
    means = []

    for i_episode in range(constants.max_episodes):

        train = True
        if (i_episode + 1) % constants.EVAL_INTERVAL == 0:
            train = False

        observation = env.reset()  #
        agent.adjust_exploration(i_episode)
        agent.adjust_lr(i_episode)

        state = quantizator.digitize(observation)
        action = agent.choose_action(state, train=train)

        for step in count():  # range(constants.max_steps):  # consider ending only on fail ?
            # env.render()
            observation, reward, done, info = env.step(action)  # takes the specified action
            if done:
                pos = observation[0]
                rot = observation[2]
                if train:
                    train_durations[i_episode] = (step + 1)
                else:
                    eval_durations[i_episode] = (step + 1)

                plot_rewards(train_durations, eval_durations)
                if pos < -2.4 or pos > 2.4:
                    print("Terminated due to position")
                # print("Episode {} terminated after {} timesteps".format(i_episode, step + 1))
                break

            new_state = quantizator.digitize(observation)
            new_action = agent.choose_action(new_state, train=train)

            agent.update(state, action, reward, new_state, new_action)  # if q-learning new action is not going to be used

            state = new_state
            action = new_action
        else:
            pass
            # print("Episode {} finished successful!".format(i_episode))

    env.close()

    plt.show()
