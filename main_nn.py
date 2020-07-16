import torch
import torch.optim as optim
import gym
from utils import constants
from utils.memory import Memory, MemoryPER
import matplotlib.pyplot as plt
import numpy as np
import logging.config
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from utils.functions import plot_rewards, plot_epsilon, check_termination
from torch.optim.lr_scheduler import ReduceLROnPlateau

TARGET_UPDATE = 100  # target net is updated with the weights of policy net once every 100 updates
BATCH_SIZE = 32

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

writer = None
if constants.TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

env = gym.make(constants.environment)
num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n

lr = 1e-3
layers_dim = [24, 48]
dropout = 0
gamma = 1
eps_decay, eps_start, eps_end = 0.001, 1, 0
mem_size = 10_000

dueling = True  # Classic and Dueling DQN architectures are supported
if dueling:
    dqn_arch = Dueling

else:
    dqn_arch = ClassicDQN

network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout)

log.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)
# scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=20)  # not used in update for now
# ExponentialLR(optimizer, lr_deacy)  # alternative scheduler
# scheduler will reduce the lr by the specified factor when metric has stopped improving
per = True
if per:
    memory = MemoryPER(mem_size)
else:
    memory = Memory(mem_size)

double = True
if double:
    agent = DoubleDQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)
else:
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)

steps_done = 0
train_rewards, eval_rewards = {}, {}
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

    episode_duration = 0
    episode_reward = 0
    total_loss = 0
    while not done:
        episode_duration += 1
        # if not train:
        #     env.render()
        action = agent.choose_action(state, train=train)
        next_state, reward, done, _ = env.step(action)  # maybe training can run in parallel with sleep
        next_state = np.float32(next_state)
        memory.store(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state
        episode_reward += reward

        if train:
            steps_done += 1
            try:
                transitions, indices, is_weights = memory.sample(BATCH_SIZE)
            except ValueError:
                continue
            loss, errors = agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net
            total_loss += loss
            agent.adjust_exploration(steps_done)  # rate is updated at every step - taken from the tutorial
            memory.batch_update(indices, errors)
            if double and (steps_done % TARGET_UPDATE == 0):  # Update the target network, had crucial impact
                agent.update_target_net()
                if constants.TENSORBOARD:
                    for name, param in agent.target_net.named_parameters():
                        headline, title = name.rsplit(".", 1)
                        writer.add_histogram('TargetNet/' + headline + '/' + title, param, i_episode)
                    writer.flush()

    if train:
        train_rewards[i_episode] = episode_reward
        if constants.TENSORBOARD:
            # writer.add_scalars('Overview/Rewards', {'Train': episode_reward}, i_episode)
            writer.add_scalar('Agent/Loss', total_loss / episode_duration, i_episode)
            writer.add_scalar('Agent/Reward Train', episode_reward, i_episode)
            writer.flush()
            for name, param in agent.policy_net.named_parameters():
                headline, title = name.rsplit(".", 1)
                writer.add_histogram('PolicyNet/' + headline + '/' + title, param, i_episode)
            writer.flush()

    else:
        eval_rewards[i_episode] = episode_reward
        if constants.TENSORBOARD:
            # writer.add_scalars('Overview/Rewards', {'Eval': episode_reward}, i_episode)
            writer.add_scalar('Agent/Reward Eval', episode_reward, i_episode)
            writer.flush()
            if check_termination(eval_rewards):
                log.info('Solved after {} episodes.'.format(len(train_rewards)))
                break

    # plot_durations(train_durations, eval_durations)
    epsilon.append(agent.epsilon)
    # plot_epsilon(epsilon)
    if constants.TENSORBOARD:
        writer.add_scalar('Agent/Epsilon', agent.epsilon, i_episode)
        writer.add_scalar('Agent/Steps', steps_done, i_episode)
        writer.flush()

else:
    log.info("Unable to reach goal in {} training episodes.".format(len(train_rewards)))

figure = plot_rewards(train_rewards, eval_rewards, completed=True)
# plt.show()
if constants.TENSORBOARD:
    writer.add_figure('Plot', figure)

    state = np.float32(env.reset())
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))

    # first dict with hparams, second dict with metrics
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Double': double, 'Dueling': dueling,
                        'PER': per, 'Mem Size': mem_size, 'Target_upd_interval': TARGET_UPDATE,
                        'Batch Size': BATCH_SIZE, 'EPS_DECAY': eps_decay},
                       {'episodes_needed': len(train_rewards)})
    writer.flush()
    writer.close()

env.close()
