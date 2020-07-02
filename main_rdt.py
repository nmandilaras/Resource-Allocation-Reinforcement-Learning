import torch
import torch.optim as optim
import numpy as np
import ast
from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser, config_parser
from nn.policy_fc import PolicyFC
from nn.dqn_archs import ClassicDQN, Dueling
from utils.memory import Memory, MemoryPER
from agents.dqn_agents import DQNAgent, DoubleDQNAgent
from torch.utils.tensorboard import SummaryWriter
from utils.config_constants import *
from utils.functions import write_metrics
from utils.constants import EPS_END


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

comment = "_{}_{}".format(config_env[BE_NAME], args.comment)
writer = SummaryWriter(comment=comment)

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
arch = config_agent[ARCH]  # Classic and Dueling DQN architectures are supported
algo = config_agent[ALGO]  # DDQN or DQN
mem_type = config_agent[MEM_PER]
mem_size = int(config_agent[MEM_SIZE])
eps_decay = float(config_agent[EPS_DECAY])

if arch:
    dqn_arch = Dueling
else:
    dqn_arch = ClassicDQN

network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout=0)

log.info("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)

if mem_type == 'per':
    memory = MemoryPER(mem_size)
else:
    memory = Memory(mem_size)

if algo == 'double':
    agent = DoubleDQNAgent(num_of_actions, network, criterion, optimizer, gamma=gamma, eps_decay=eps_decay)
else:
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, gamma=gamma, eps_decay=eps_decay)

done = False
step = 0
total_reward = 0
exploration_viol = 0
end_exploration_step = 0
end_exploration_flag = False

try:
    state = env.reset()
    state = np.float32(state)

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.float32(next_state)
        memory.store(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        if agent.epsilon < EPS_END + 0.01 and not end_exploration_flag:
            log.info("Conventional end of exploration at step: {}".format(step))
            exploration_viol = env.violations
            env.violations = 0
            end_exploration_step = step
            end_exploration_flag = True

        step += 1
        total_reward += reward
        try:
            transitions, indices, is_weights = memory.sample(batch_size)
        except ValueError:
            continue
        loss, errors = agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net
        agent.adjust_exploration(step)  # rate is updated at every step
        memory.batch_update(indices, errors)
        if step % target_update == 0:  # Update the target network, had crucial impact
            agent.update_target_net()
            log_net(agent.target_net, 'TargetNet', step)

        for key, value in info.items():
            write_metrics(key, value, writer, step)
        writer.add_scalar('Agent/Action', action, step)
        writer.add_scalar('Agent/Reward', reward, step)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, step)
        writer.add_scalar('Agent/Loss', loss, step)
        writer.flush()
        log_net(agent.policy_net, 'PolicyNet', step)

    log.info("Be finished")
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Target_upd_interval': target_update,
                         'Double': algo, 'Dueling': arch, 'Batch Size': batch_size, 'Mem PER': mem_type,
                        'Mem Size': mem_size},
                       {'violations': (env.violations - exploration_viol) / (step - end_exploration_step),
                        'violations_total': env.violations / step,
                        'slow_down': env.interval_bes})
finally:
    writer.flush()
    writer.close()
    env.stop()
