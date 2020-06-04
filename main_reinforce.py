import gym
import logging.config
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import constants
from agents.reinforce_agent import Reinforce
from utils.functions import plot_rewards, check_termination
from nn.policy_fc import PolicyFC

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')

writer = None
if constants.TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

env = gym.make(constants.environment)

# seed = 543  # use for reproducibility
# env.seed(seed)
# torch.manual_seed(seed)

num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n
train_rewards, eval_durations = {}, {}

lr = 1e-2  # from pytorch tutorial and others
layers_dim = [6]
gamma = 0.999

network = PolicyFC(num_of_observations, layers_dim, num_of_actions)

logger.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

optimizer = optim.Adam(network.parameters(), lr)

agent = Reinforce(env.action_space.n, network, optimizer, gamma)

for i_episode in range(constants.max_episodes):
    log_probs, rewards, max_probs = [], [], []

    next_state = env.reset()

    done = False
    train = True
    agent.train_mode()
    if (i_episode + 1) % constants.EVAL_INTERVAL == 0:
        train = False
        agent.eval_mode()

    episode_duration = 0
    while not done:
        episode_duration += 1
        # if not train:
        #     env.render()
        state = np.float32(next_state)
        action, log_prob, max_prob = agent.choose_action(state, train=train)  # TODO merge train parameter with model_train
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)  # even if episode is done we keep the reward and log prop, is this a problem?
        rewards.append(reward)
        max_probs.append(max_prob)  # only needed for monitoring

    episode_reward = sum(rewards)
    if train:
        train_rewards[i_episode] = episode_reward
        discounted_rewards = agent.calculate_rewards(rewards)
        loss = agent.update(log_probs, discounted_rewards)
        if constants.TENSORBOARD:
            writer.add_scalars('Overview/Rewards', {'Train': episode_reward}, i_episode)
            writer.add_scalar('Overview/Loss', loss, i_episode)
            writer.add_scalar('Reward/Train', episode_reward, i_episode)
            writer.add_scalar('Probs/Train', sum(max_probs) / len(max_probs), i_episode)
            writer.flush()
            for name, param in agent.policy_net.named_parameters():
                headline, title = name.rsplit(".", 1)
                writer.add_histogram(headline + '/' + title, param, i_episode)
            writer.flush()

    else:
        eval_durations[i_episode] = episode_reward
        if constants.TENSORBOARD:
            writer.add_scalars('Overview/Rewards', {'Eval': episode_reward}, i_episode)
            writer.add_scalar('Reward/Eval', episode_reward, i_episode)
            writer.add_scalar('Probs/Eval', sum(max_probs) / len(max_probs), i_episode)
            writer.flush()
        if check_termination(eval_durations):
            logger.info('Solved after {} episodes.'.format(len(train_rewards)))
            break

    # plot_durations(train_durations, eval_durations)
else:
    logger.info("Unable to reach goal in {} training episodes.".format(len(train_rewards)))

figure = plot_rewards(train_rewards, eval_durations, completed=True)


if constants.TENSORBOARD:
    writer.add_figure('Plot', figure)

    state = np.float32(env.reset())
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))

    # first dict with hparams, second dict with metrics
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'Hidden Layers Dims': str(layers_dim)},
                       {'episodes_needed': len(train_rewards)})
    writer.flush()

    writer.close()
else:
    plt.show()

env.close()
