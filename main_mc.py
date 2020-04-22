import gym
from utils import constants
from agents.classic_agents.mc_agent import MCAgent
import logging.config
from utils.quantization import Quantization
from utils.functions import plot_rewards, check_termination
import matplotlib.pyplot as plt

# COMMENT it seems that monte carlo has high variance maybe we should reduce exploration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

env = gym.make(constants.environment)
train_durations = {}
eval_durations = {}

num_of_actions = env.action_space.n

high_intervals = env.observation_space.high
low_intervals = env.observation_space.low

vars_ls = list(zip(low_intervals, high_intervals, constants.var_freq))
quantizator = Quantization(vars_ls, lambda x: [x[i] for i in [0, 1, 2, 3]])

agent = MCAgent(num_of_actions, quantizator.dimensions)

for i_episode in range(constants.max_episodes):
    # Initialize the environment and state
    done = False
    train = True
    if (i_episode + 1) % constants.EVAL_INTERVAL == 0:
        train = False

    next_observation = env.reset()
    agent.adjust_exploration(i_episode)
    state_action_ls = []
    reward_ls = []

    t = 0
    while not done:
        t += 1
        # env.render()
        state = quantizator.digitize(next_observation)
        action = agent.choose_action(state, train=train)    # Select and perform an action
        next_observation, reward, done, _ = env.step(action)

        state_action_ls.append((state, action))
        reward_ls.append(reward)

    if train:
        train_durations[i_episode] = (t + 1)
        discounted_rewards = agent.calculate_rewards(reward_ls)
        agent.update(state_action_ls, discounted_rewards)
    else:
        eval_durations[i_episode] = (t + 1)
        if check_termination(eval_durations):
            logger.info('Solved after {} episodes.'.format(len(train_durations)))
            break
    plot_rewards(train_durations, eval_durations)

else:
    logger.info("Unable to reach goal in {} training episodes.".format(len(train_durations)))

plot_rewards(train_durations, eval_durations, completed=True)
env.close()
plt.show()
