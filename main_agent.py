import torch
import torch.optim as optim
import numpy as np
import ast
from rlsuite.builders.factories import memory_factory
from env_builder import EnvBuilder
import logging.config
from rlsuite.utils.functions import log_parameters_histograms
from rlsuite.builders.agent_builder import DQNAgentBuilder
from torch.utils.tensorboard import SummaryWriter
from utils.config_constants import *
from utils.constants import Loaders, Schedulers
from utils.functions import write_metrics, form_duration, config_parser
from utils.argparser import cmd_parser
from datetime import datetime
import os

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

MEM_START_SIZE = 1000

time_at_start = datetime.now().strftime('%b%d_%H-%M-%S')
parser = cmd_parser()
args = parser.parse_args()

config = config_parser(args.config_file)

# some arguments are set from command line args, that was useful for tuning
if config[LOADER][ACTION_INTERVAL] == "-1":
    config[LOADER][ACTION_INTERVAL] = args.interval

if config[AGENT][EPS_DECAY] == "-1":
    config[AGENT][EPS_DECAY] = args.decay

config[LOADER][QUANTILE] = args.quantile
config[ENV][FEATURE] = args.feature

env = EnvBuilder() \
    .build_pqos(config[PQOS][PQOS_INTERFACE], config[PQOS][CORES_LC], config[SCHEDULER][CORES_BE]) \
    .build_loader(Loaders.MEMCACHED, config[LOADER]) \
    .build_scheduler(Schedulers.QUEUE, config[SCHEDULER]) \
    .build(config[ENV])

comment = "_{}".format(args.comment)
writer = SummaryWriter(comment=comment)

num_of_observations = env.observation_space.shape[0]
num_of_actions = env.action_space.n

log.info("Number of available actions: {}".format(num_of_actions))
log.info("NUmber of input features: {}".format(num_of_observations))

# TODO handle this in an elegant way, maybe we good use a dict that maps each field to a function that can be applied
#   in order to ge the type.
lr = config[AGENT].getfloat(LR)
layers_dim = ast.literal_eval(config[AGENT][LAYERS_DIM])
target_update = config[AGENT].getint(TARGET_UPDATE)
batch_size = config[AGENT].getint(BATCH_SIZE)
arch = config[AGENT][ARCH]  # Vanilla or Dueling DQN
agent_algorithm = config[AGENT][ALGO]  # DDQN or DQN
mem_type = config[AGENT][MEM_PER]
mem_size = config[AGENT].getint(MEM_SIZE)
gamma = config[AGENT].getfloat(GAMMA)
eps_decay = config[AGENT].getfloat(EPS_DECAY)
eps_start = config[AGENT].getfloat(EPS_START)
eps_end = config[AGENT].getfloat(EPS_END)
checkpoint_path = config[AGENT][CHECKPOINT]
init_weights = config[AGENT][WEIGHTS]

criterion = torch.nn.MSELoss(reduction='none')  # torch.nn.SmoothL1Loss()  # Huber loss
optimizer = optim.Adam

memory = memory_factory(mem_type, mem_size)

agent = DQNAgentBuilder(num_of_observations, num_of_actions, gamma, eps_decay, eps_start, eps_end) \
    .set_criterion(criterion) \
    .build_network(layers_dim, arch) \
    .load_checkpoint(checkpoint_path) \
    .build_optimizer(optimizer, lr) \
    .build(agent_algorithm)

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
        # start_step_time = time.time()
        # could run in parallel with the rest of the loop but GIL prevents this
        next_state, reward, done, info = env.step(action)
        # end_step_time = time.time()
        # step_interval = (end_step_time - start_step_time) * 1000
        # writer.add_scalar('Timing/Env Step', step_interval, step)
        next_state = np.float32(next_state)
        memory.store(state, action, next_state, reward, done)  # Store the transition in memory
        state = next_state

        step += 1

        if mem_type == 'per' and memory.tree.n_entries < MEM_START_SIZE:
            continue

        # measure the violations of the exploration phase separately
        if agent.epsilon < eps_end + 0.01 and not end_exploration_flag:
            log.info("Conventional end of exploration at step: {}".format(step))
            exploration_viol = env.violations
            end_exploration_step = step
            end_exploration_flag = True

        total_reward += reward

        # experimental path used to create checkpoints: increase exploration when new be is started
        # if new_be:
        #     log.info("New be started at step: {}. Exploration rate increased.".format(step))
        #     decaying_schedule = min(decaying_schedule, 0)  # resets exploration rate at 0.2 with 3210, 4500 for 0.1
        #     # memory.flush()  # we didn't observe any benefit from emptying the memory
        #
        #     save_file = os.path.join('checkpoints', time_at_start + comment + '_' + str(step) + '.pkl')
        #     agent.save_checkpoint(save_file)

        try:
            transitions, indices, is_weights = memory.sample(batch_size)
        except ValueError:  # not enough samples in memory
            continue

        decaying_schedule += 1

        loss, errors = agent.update(transitions, is_weights)  # Perform one step of optimization on the policy net
        agent.adjust_exploration(decaying_schedule)  # rate is updated at every step
        memory.batch_update(indices, errors)  # only applicable for per

        if step % target_update == 0:  # Update the target network
            agent.update_target_net()
            # creates enormous amount of data and gives little information so we disable the logging of weights
            # log_parameters_histograms(writer, agent.target_net, step, 'TargetNet')

        for key, value in info.items():
            write_metrics(writer, key, value, step)
        writer.add_scalar('Agent/Action', action, step)
        writer.add_scalar('Agent/Reward', reward, step)
        writer.add_scalar('Agent/Reward Cumulative', total_reward, step)
        writer.add_scalar('Agent/Epsilon', agent.epsilon, step)
        writer.add_scalar('Agent/Loss', loss, step)
        writer.flush()
        # log_parameters_histograms(writer, agent.policy_net, step, 'PolicyNet')

        # measuring training time
        # end_training = time.time()
        # training_interval = (end_training - end_step_time) * 1000
        # writer.add_scalar('Timing/Training', training_interval, step)

    log.info("Experiment finished after {} steps.".format(step))
    duration = env.get_experiment_duration()
    writer.add_graph(agent.policy_net, torch.tensor(state, device=agent.device))
    writer.add_hparams({'lr': lr, 'gamma': gamma, 'HL Dims': str(layers_dim), 'Target_upd_interval': target_update,
                        'Algorithm': agent_algorithm, 'Arch': arch, 'Batch Size': batch_size, 'Mem Type': mem_type,
                        'Mem Size': mem_size},
                       {'Results/Viol. Post-Expl.': (env.violations - exploration_viol) / (step - end_exploration_step),
                        'Results/Viol. Exploration': exploration_viol / end_exploration_step,
                        'Results/Violations Total': env.violations / step,
                        'Results/Time': duration})

    writer.add_text('duration', form_duration(duration))

finally:
    save_file = os.path.join('checkpoints', time_at_start + comment + '.pkl')
    agent.save_checkpoint(save_file)

    writer.flush()
    writer.close()
    env.stop()
