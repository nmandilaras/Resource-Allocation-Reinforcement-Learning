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
from datetime import datetime
import os
import time

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

def log_net(net, net_name, step):
    """"""
    for name, param in net.named_parameters():
        headline, title = name.rsplit(".", 1)
        writer.add_histogram(net_name + '/' + headline + '/' + title, param, step)
    writer.flush()


time_at_start = datetime.now().strftime('%b%d_%H-%M-%S')
parser = cmd_parser()
args = parser.parse_args()

config_env, config_agent, config_misc = config_parser(args.config_file)

if config_env[ACTION_INTERVAL] == "-1":
    config_env[ACTION_INTERVAL] = args.interval

if config_env[BE_NAME] == 'multi':
    config_env[BE_NAME] = str([args.be_name])

if config_agent[EPS_DECAY] == "-1":
    config_agent[EPS_DECAY] = args.decay

config_env[QUANTILE] = args.quantile
config_env[FEATURE] = args.feature

comment = "_{}".format(args.comment)
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
eps_start = float(config_agent[EPS_START])
eps_end = float(config_agent[EPS_END])
checkpoint_path = config_agent[CHECKPOINT]
init_weights = config_agent[WEIGHTS]

if arch == 'dueling':
    log.info('Dueling architecture will be used.')
    dqn_arch = Dueling
else:
    dqn_arch = ClassicDQN

network = PolicyFC(num_of_observations, layers_dim, num_of_actions, dqn_arch, dropout=0)

if checkpoint_path:
    log.info("Loading weights from checkpoint.")
    weights = torch.load(checkpoint_path)

    # for var_name in weights:
    #     print(var_name, "\t", weights[var_name].size())

    if init_weights == 'init':
        log.info("Weights of last layers will be reinitialized.")

        weights["output.value_stream.0.weight"] = torch.rand(1, weights["output.value_stream.0.weight"].size(1), requires_grad=True)
        weights["output.value_stream.0.bias"] = torch.rand(1, requires_grad=True)

        weights["output.advantage_stream.0.weight"] = torch.rand(num_of_actions, weights["output.advantage_stream.0.weight"].size(1), requires_grad=True)
        weights["output.advantage_stream.0.bias"] = torch.rand(num_of_actions, requires_grad=True)

    network.load_state_dict(weights)

criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam(network.parameters(), lr)

if mem_type == 'per':
    log.info('Prioritized Experience Replay will be used.')
    memory = MemoryPER(mem_size)
else:
    memory = Memory(mem_size)

if algo == 'double':
    log.info('Double DQN Agent will be used.')
    agent = DoubleDQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)
else:
    agent = DQNAgent(num_of_actions, network, criterion, optimizer, gamma, eps_decay, eps_start, eps_end)

log.info("Number of parameters in our model: {}".format(sum(x.numel() for x in network.parameters())))

done = False
step = 0
decaying_schedule = 0
total_reward = 0
exploration_viol = 0
end_exploration_step = 1
end_exploration_flag = False

try:
    state = env.reset()
    state = np.float32(state)

    while not done:
        action = agent.choose_action(state)
        # measuring env step time
        # start_time = time.time()
        next_state, reward, done, info, new_be = env.step(action)  # could run in parallel with the rest of the loop but GIL prevents this
        # end_time = time.time()
        # time_interval = (end_time - start_time) * 1000
        # writer.add_scalar('Timing/Env Step', time_interval, step)
        next_state = np.float32(next_state)
        memory.store(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        step += 1
        if mem_type == 'per' and memory.tree.n_entries < 1000:
            continue

        # measure the violations of the exploration phase separately
        if agent.epsilon < eps_end + 0.01 and not end_exploration_flag:
            log.info("Conventional end of exploration at step: {}".format(step))
            exploration_viol = env.violations
            end_exploration_step = step
            end_exploration_flag = True

        # step += 1
        total_reward += reward

        # use for online training
        if new_be:
            log.info("New be started at step: {}. Exploration rate increased.".format(step))
            decaying_schedule = min(decaying_schedule, 0)  # resets exploration rate at 0.2 with 3210, 4500 for 0.1
            log.info("Memory was flushed.")
            memory.flush()

            save_file = os.path.join('checkpoints', time_at_start + comment + '_' + str(step) + '.pkl')
            torch.save(agent.policy_net.state_dict(), save_file)

        try:
            transitions, indices, is_weights = memory.sample(batch_size)
        except ValueError:
            continue

        decaying_schedule += 1

        loss, errors = agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net

        agent.adjust_exploration(decaying_schedule)  # rate is updated at every step
        memory.batch_update(indices, errors)
        if step % target_update == 0:  # Update the target network, had crucial impact
            agent.update_target_net()
            log_net(agent.target_net, 'TargetNet', step)

        for key, value in info.items():
            write_metrics(key, value, writer, step)
        writer.add_scalar('Agent/Action', action, step)
        writer.add_scalar('Agent/Reward', reward, step)
        writer.add_scalar('Agent/Reward Cumulative', total_reward, step)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, step)
        writer.add_scalar('Agent/Loss', loss, step)
        writer.flush()
        log_net(agent.policy_net, 'PolicyNet', step)

        # measuring training time
        # end_time_2 = time.time()
        # time_interval_2 = (end_time_2 - end_time) * 1000
        # writer.add_scalar('Timing/Training', time_interval_2, step)

    log.info("Experiment finished after {}".format(step))
    minutes = int(env.interval_bes)
    seconds = int(round((env.interval_bes % 1) * 60, 0))
    duration = str(minutes) + 'm' + str(seconds) + 's'
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Target_upd_interval': target_update,
                         'Double': algo, 'Dueling': arch, 'Batch Size': batch_size, 'Mem PER': mem_type,
                        'Mem Size': mem_size},
                       {'Results/Violations': (env.violations - exploration_viol) / (step - end_exploration_step),
                        'Results/Violations Exploration': exploration_viol / end_exploration_step,
                        'Results/Violations Total': env.violations / step,
                        'Results/Time': env.interval_bes})

    writer.add_text('duration', duration)

finally:
    save_file = os.path.join('checkpoints', time_at_start + comment + '.pkl')
    torch.save(agent.policy_net.state_dict(), save_file)

    writer.flush()
    writer.close()
    env.stop()
