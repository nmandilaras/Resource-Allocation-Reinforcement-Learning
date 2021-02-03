from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser
from rlsuite.utils.functions import config_parser
from torch.utils.tensorboard import SummaryWriter
from utils.functions import write_metrics
from utils.constants import LC_TAG, BE_TAG
from utils.config_constants import *
import time
import random


def monitor_warm_up():
    global step, start_time
    for i in range(args.warm_up):
        q95_latency, rps = env.get_loader_stats()
        env.update_hw_metrics()
        end_time = time.time()
        time_interval = end_time - start_time
        start_time = end_time
        ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps = env.get_lc_metrics(time_interval)
        write_metrics(LC_TAG, (ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps, q95_latency, rps), writer, step)
        step += 1


step = 0

parser = cmd_parser()
parser.add_argument('--warm-up', type=int, default=0, help='Time to collect metrics before/after bes execution in sec')
parser.add_argument('--ways-be', type=int, default=-1, help='Ways to be allocated to best effort group')
args = parser.parse_args()

config_env, config_agent, config_misc = config_parser(args.config_file)

if config_env[BES_LIST] == 'multi':
    config_env[BES_LIST] = str([args.be_name])

config_env[QUANTILE] = args.quantile
config_env[FEATURE] = args.feature

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

comment = "_measurement_action_{}_{}".format(args.ways_be, args.comment)
writer = SummaryWriter(comment=comment)

env = Rdt(config_env)

try:
    env.reset_pqos()
    env.set_association_class(args.ways_be)
    #time.sleep(5)
    start_time = time.time()
    env.start_client()
    log.debug("Mem client started. Warm up period follows.")

    # collect tail latency and hw metrics before launching be
    monitor_warm_up()

    env.generator = random.Random(env.seed)
    env.start_bes()

    log.info("Num of ways that are going to be statically allocated to BEs: {}".format(args.ways_be))

    done = False

    while not done:
        done = env.determine_termination() if env.steps % (1000 // int(env.action_interval)) == 0 else False

        q95_latency, rps = env.get_loader_stats()
        env.update_hw_metrics()
        end_time = time.time()
        time_interval = end_time - start_time
        start_time = end_time
        reward = env.reward_func(args.ways_be, q95_latency)
        ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps, cycles_hp, instructions_hp = env.get_lc_metrics(time_interval)
        ipc_be, misses_be, llc_be, mbl_be_ps, mbr_be_ps, cycles_be, instructions_be = env.get_be_metrics(time_interval)

        misses_be = misses_be / (cycles_be / 1000.)
        misses_hp = misses_hp / (cycles_be / 1000.)

        write_metrics(LC_TAG, (ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps, q95_latency, rps), writer, step)
        write_metrics(BE_TAG, (ipc_be, misses_be, llc_be, mbl_be_ps, mbr_be_ps, None), writer, step)

        # next_state, reward, done, info = env.step(args.ways_be)
        # writer.add_scalar('Agent/Action', args.ways_be, step)
        writer.add_scalar('Agent/Reward', reward, step)

        step += 1
        env.steps += 1

    log.info("Be finished")
    writer.add_hparams({'Action': args.ways_be},
                       {'Results/Violations Total': env.violations / step, 'Results/Time': env.interval_bes})

    minutes = int(env.interval_bes)
    seconds = int(round((env.interval_bes % 1) * 60, 0))
    duration = str(minutes) + 'm' + str(seconds) + 's'
    writer.add_text('duration', duration)

    writer.flush()

    log.info('Percentage of violations: {}'.format(env.violations / env.steps))
    log.info('Duration of experiment: {}m{}s'.format(minutes, seconds))

finally:
    log.warning('Stopping everything!')
    env.stop_bes()

    monitor_warm_up()

    writer.flush()
    writer.close()
    env.stop_client()
    env.stop_pqos()
