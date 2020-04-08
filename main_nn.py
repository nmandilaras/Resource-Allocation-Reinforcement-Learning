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
from agents.dqn_agents import DQNAgent, DDQNAgent
from utils.functions import plot_durations, plot_epsilon, check_termination
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

TARGET_UPDATE = 10  # target net is updated with the weights of policy net once every 10 episodes
BATCH_SIZE = 32

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

writer = None
if constants.TENSORBOARD:
    writer = SummaryWriter()

env = gym.make(constants.environment)

arch = DQNArch.CLASSIC  # Classic and Dueling DQN architectures are supported
if arch == DQNArch.CLASSIC:
    dqn_arch = ClassicDQN
elif arch == DQNArch.DUELING:
    dqn_arch = Dueling
else:
    raise NotImplementedError

num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n

lr = 1e-2
layers_dim = [24, 48]
dropout = 0
gamma = 1
mem_size = 100_000

network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout)

logger.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)  # TODO call scheduler from update function
# ExponentialLR(optimizer, lr_deacy)  # alternative scheduler
# scheduler will reduce the lr by the specified factor when metric has stopped improving
memory = Memory(mem_size)  # single transition is used in vanilla update

double = False
if double:
    agent = DDQNAgent(num_of_actions, network, criterion, optimizer, gamma)
else:
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, gamma)

steps_done = 0
train_durations, eval_durations = {}, {}
epsilon = []


for i_episode in range(constants.max_episodes):
    # TODO should i start with a full memory ?
    # TODO memory should be episoditic or not ??
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

    if double and ((i_episode + 1) % TARGET_UPDATE == 0):  # Update the target network
        agent.update_target_net()

        if constants.TENSORBOARD:
            for name, param in agent.target_net.named_parameters():
                headline, title = name.rsplit(".", 1)
                writer.add_histogram('TargetNet/' + headline + '/' + title, param, i_episode)
            writer.flush()
        # agent.memory.flush()

else:
    logger.info("Unable to reach goal in {} training episodes.".format(len(train_durations)))

figure = plot_durations(train_durations, eval_durations, completed=True)
# plt.show()
if constants.TENSORBOARD:
    writer.add_figure('Plot', figure)

    state = np.float32(env.reset())
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))

    # writer.add_text('Parameters', 'Optimizer used: Adam')
    # layout = {'Overview': {'Reward': ['Multiline', ['Reward/Train', 'Reward/Eval']]}}
    # writer.add_custom_scalars(layout)

    # first dict with hparams, second dict with metrics
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'Hidden Layers Dims': str(layers_dim)},
                  {'episodes_needed': len(train_durations)})
    writer.flush()
    writer.close()

env.close()
