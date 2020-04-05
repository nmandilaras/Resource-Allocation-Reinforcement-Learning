import torch
import torch.optim as optim
import gym
from utils import constants
from utils.constants import DQNArch
import matplotlib.pyplot as plt
import numpy as np
import logging.config
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from agents.double_dqn_agent import DDQNAgent
from agents.dqn_agent import DQNAgent
from itertools import count
from utils.functions import plot_durations, plot_epsilon, check_termination

TARGET_UPDATE = 10  # target net is updated with the weights of policy net once every 10 episodes

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

env = gym.make(constants.environment)

arch = DQNArch.DUELING  # Classic, Dueling DQN architectures are supported
# choose architecture
if arch == DQNArch.CLASSIC:
    dqn_arch = ClassicDQN
elif arch == DQNArch.DUELING:
    dqn_arch = Dueling
else:
    raise NotImplementedError

network = PolicyFC(env.observation_space.shape[0], [64, 64*2], env.action_space.n, dqn_arch)

criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr=1e-2)
mem_size = 1000

double = True
# choose agent
if double:
    agent = DDQNAgent(env.action_space.n, network, criterion, optimizer, mem_size)
else:
    agent = DQNAgent(env.action_space.n, network, criterion, optimizer, mem_size)

steps_done = 0
train_durations, eval_durations = {}, {}
epsilon = []
means = []


for i_episode in range(constants.max_episodes):
    # TODO memory should be episoditic or not ??
    # TODO plot epsilon per episode
    # Initialize the environment and state
    state = env.reset()
    state = np.float32(state)
    done = False
    train = True
    agent.train_mode()
    if (i_episode + 1) % constants.EVAL_INTERVAL == 0:
        train = False
        agent.eval_mode()

    t = 0
    while not done:
        t += 1
        # env.render()
        action = agent.choose_action(state, train=train)
        next_state, reward, done, _ = env.step(action)
        next_state = np.float32(next_state)
        reward = torch.tensor([reward])

        if done:
            next_state = None

        agent.push_in_memory(state, action, next_state, reward)  # Store the transition in memory
        state = next_state

        if train:
            steps_done += 1
            agent.update()  # Perform one step of the optimization (on the policy network)
            agent.adjust_exploration(steps_done)  # rate is updated at every step - taken from the tutorial

    if train:
        train_durations[i_episode] = (t + 1)
    else:
        eval_durations[i_episode] = (t + 1)
        if check_termination(eval_durations):
            logger.info('Solved after {} episodes.'.format(len(train_durations)))
            break

    plot_durations(train_durations, eval_durations)
    epsilon.append(agent.epsilon)
    plot_epsilon(epsilon)


    if (i_episode + 1) % TARGET_UPDATE == 0:  # Update the target network
        agent.update_target_net()
        # agent.memory.flush()

else:
    logger.info("Unable to reach goal in {} training episodes.".format(len(train_durations)))

plot_durations(train_durations, eval_durations, completed=True)
env.close()
plt.show()
