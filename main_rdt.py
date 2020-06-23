import torch
import torch.optim as optim
import numpy as np
import ast
from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser, config_parser
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from utils.memory import Memory
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from torch.utils.tensorboard import SummaryWriter
from utils.config_constants import *


def log_net(net, net_name, step):
    """"""
    for name, param in net.named_parameters():
        headline, title = name.rsplit(".", 1)
        writer.add_histogram(net_name + '/' + headline + '/' + title, param, step)
    writer.flush()


def write_metrics(tag, metrics):
    ipc, misses, llc, mbl, mbr, latency = metrics
    if tag == 'Latency Critical':
        writer.add_scalar('Latency Critical/Latency', latency, step)
    header = '{}/'.format(tag)
    writer.add_scalar(header + 'IPC', ipc, step)
    writer.add_scalar(header + 'Misses', misses, step)
    writer.add_scalar(header + 'LLC', llc, step)
    writer.add_scalar(header + 'MBL', mbl, step)
    writer.add_scalar(header + 'MBR', mbr, step)
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

log.info("Number of available actions: {}".format(num_of_actions))
log.info("NUmber of input features: {}".format(num_of_observations))

lr = float(config_agent[LR])
layers_dim = ast.literal_eval(config_agent[LAYERS_DIM])
target_update = int(config_agent[TARGET_UPDATE])  # target net is updated with the weights of policy net every x updates
batch_size = int(config_agent[BATCH_SIZE])
gamma = float(config_agent[GAMMA])
mem_size = int(config_agent[MEM_SIZE])

dqn_arch = Dueling
network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout=0)

log.info("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

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
        next_state, reward, done, info = env.step(action)
        next_state = np.float32(next_state)
        memory.push(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        step += 1
        try:
            transitions = memory.sample(batch_size)
        except ValueError:
            continue
        loss = agent.update(transitions)  # Perform one step of the optimization (on the policy network)
        agent.adjust_exploration(step)  # rate is updated at every step - taken from the tutorial
        if step % target_update == 0:  # Update the target network, had crucial impact
            agent.update_target_net()
            log_net(agent.target_net, 'TargetNet', step)

        for key, value in info.items():
            write_metrics(key, value)
        writer.add_scalar('Agent/Action', action, step)
        writer.add_scalar('Agent/Reward', reward, step)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, step)
        writer.add_scalar('Agent/Loss', loss, step)
        writer.flush()
        log_net(agent.policy_net, 'PolicyNet', step)

    log.info("Be finished")
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Target_upd_interval': target_update,
                        'Batch Size': batch_size}, {})
    writer.flush()
finally:
    env.stop()
