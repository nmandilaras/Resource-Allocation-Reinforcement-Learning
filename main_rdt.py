import torch
import torch.optim as optim
import numpy as np
from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser, config_parser
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from utils.memory import Memory
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from torch.utils.tensorboard import SummaryWriter


def log_net(net, net_name, step):
    """"""
    for name, param in net.named_parameters():
        headline, title = name.rsplit(".", 1)
        writer.add_histogram(net_name + '/' + headline + '/' + title, param, step)
    writer.flush()


parser = cmd_parser()
args = parser.parse_args()
config_env, config_agent, config_misc = config_parser(args.config_file)

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

writer = SummaryWriter()

env = Rdt(config_env)

num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n

log.debug("Number of available actions: {}".format(num_of_actions))
log.debug("NUmber of input features: {}".format(num_of_observations))

TARGET_UPDATE = 100  # target net is updated with the weights of policy net once every 100 updates
BATCH_SIZE = 32

lr = 1e-3
layers_dim = [24, 48]
dropout = 0
gamma = 1
mem_size = 100_000

dqn_arch = Dueling
network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout)

log.debug("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

criterion = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)
memory = Memory(mem_size)

agent = DoubleDQNAgent(num_of_actions, network, criterion, optimizer, gamma=gamma)

done = False
step = 0

try:
    state = env.reset()
    state = np.float32(state)

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.float32(next_state)
        memory.push(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        step += 1
        try:
            transitions = memory.sample(BATCH_SIZE)
        except ValueError:
            continue
        loss = agent.update(transitions)  # Perform one step of the optimization (on the policy network)
        agent.adjust_exploration(step)  # rate is updated at every step - taken from the tutorial
        if step % TARGET_UPDATE == 0:  # Update the target network, had crucial impact
            agent.update_target_net()
            log_net(agent.target_net, 'TargetNet', step)

        writer.add_scalar('Agent/Loss', loss, step)
        writer.add_scalar('Agent/Reward', reward, step)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, step)
        log_net(agent.policy_net, 'PolicyNet', step)

    log.info("Be finished")
finally:
    env.stop()
