import torch
import torch.optim as optim
import gym
from utils import constants
from utils.constants import DQNArch
from utils.memory import Memory
import matplotlib.pyplot as plt
import numpy as np
import logging.config
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from utils.functions import plot_durations, plot_epsilon, check_termination
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

TARGET_UPDATE = 100  # target net is updated with the weights of policy net once every 100 updates
BATCH_SIZE = 32

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

writer = None
if constants.TENSORBOARD:
    writer = SummaryWriter()

env = gym.make(constants.environment)
num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n

lr = 1e-3
layers_dim = [24, 48]
dropout = 0
gamma = 1
mem_size = 100_000

dueling = True  # Classic and Dueling DQN architectures are supported
if dueling:
    dqn_arch = Dueling
else:
    dqn_arch = ClassicDQN

network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout)

logger.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=20)  # not used in update for now
# ExponentialLR(optimizer, lr_deacy)  # alternative scheduler
# scheduler will reduce the lr by the specified factor when metric has stopped improving
memory = Memory(mem_size)

double = True
if double:
    agent = DoubleDQNAgent(num_of_actions, network, criterion, optimizer, scheduler, gamma)
else:
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, scheduler, gamma)

steps_done = 0
train_durations, eval_durations = {}, {}
epsilon = []


for i_episode in range(constants.max_episodes):
    # DQN paper starts from a partially
    # memory is not episodic
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
    total_loss = 0
    while not done:
        t += 1
        # env.render()
        action = agent.choose_action(state, train=train)
        next_state, reward, done, _ = env.step(action)
        next_state = np.float32(next_state)
        memory.push(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        if train:
            steps_done += 1
            try:
                transitions = memory.sample(BATCH_SIZE)
            except:
                continue
            total_loss += agent.update(transitions)  # Perform one step of the optimization (on the policy network)
            agent.adjust_exploration(steps_done)  # rate is updated at every step - taken from the tutorial
            if double and (steps_done % TARGET_UPDATE == 0):  # Update the target network, had crusial impact
                agent.update_target_net()
                if constants.TENSORBOARD:
                    for name, param in agent.target_net.named_parameters():
                        headline, title = name.rsplit(".", 1)
                        writer.add_histogram('TargetNet/' + headline + '/' + title, param, i_episode)
                    writer.flush()

    if train:
        train_durations[i_episode] = (t + 1)
        if constants.TENSORBOARD:
            writer.add_scalars('Overview/Rewards', {'Train': t+1}, i_episode)
            writer.add_scalar('Overview/Loss', total_loss/t, i_episode)
            writer.add_scalar('Reward/Train', t + 1, i_episode)
            writer.flush()
            for name, param in agent.policy_net.named_parameters():
                headline, title = name.rsplit(".", 1)
                writer.add_histogram('PolicyNet/' + headline + '/' + title, param, i_episode)
            writer.flush()

    else:
        eval_durations[i_episode] = (t + 1)
        if constants.TENSORBOARD:
            writer.add_scalars('Overview/Rewards', {'Eval': t + 1}, i_episode)
            writer.add_scalar('Reward/Eval', t + 1, i_episode)
            writer.flush()
            if check_termination(eval_durations):
                logger.info('Solved after {} episodes.'.format(len(train_durations)))
                break

    # plot_durations(train_durations, eval_durations)
    epsilon.append(agent.epsilon)
    # plot_epsilon(epsilon)
    if constants.TENSORBOARD:
        writer.add_scalar('Epsilon', agent.epsilon, i_episode)
        writer.flush()

else:
    logger.info("Unable to reach goal in {} training episodes.".format(len(train_durations)))

figure = plot_durations(train_durations, eval_durations, completed=True)
# plt.show()
if constants.TENSORBOARD:
    writer.add_figure('Plot', figure)

    state = np.float32(env.reset())
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))

    # first dict with hparams, second dict with metrics
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Double': double, 'Dueling': dueling,
                        'Target_upd_interval': TARGET_UPDATE, 'Batch Size': BATCH_SIZE},
                  {'episodes_needed': len(train_durations)})
    writer.flush()
    writer.close()

env.close()
