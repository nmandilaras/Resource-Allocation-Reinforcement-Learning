from env_builder import EnvBuilder
import logging.config
from utils.argparser import cmd_parser
from torch.utils.tensorboard import SummaryWriter
from utils.functions import write_metrics, form_duration, config_parser
from utils.constants import Loaders, Schedulers
from utils.config_constants import *

# This script enforces static allocation and writes the metrics of the execution.

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

parser = cmd_parser()
parser.add_argument('--ways-be', type=int, default=-1, help='Ways to be allocated to best effort group')
args = parser.parse_args()

config = config_parser(args.config_file)

if config[LOADER][ACTION_INTERVAL] == "-1":
    config[LOADER][ACTION_INTERVAL] = args.interval

config[LOADER][QUANTILE] = args.quantile
config[ENV][FEATURE] = args.feature

env = EnvBuilder() \
    .build_pqos(config[PQOS][PQOS_INTERFACE], config[PQOS][CORES_LC], config[SCHEDULER][CORES_BE]) \
    .build_loader(Loaders.MEMCACHED, config[LOADER]) \
    .build_scheduler(Schedulers.QUEUE, config[SCHEDULER]) \
    .build(config[ENV])

comment = "_measurement_action_{}_{}".format(args.ways_be, args.comment)
writer = SummaryWriter(comment=comment)

done = False
log.info("Num of ways that are going to be statically allocated to BEs: {}".format(args.ways_be))

try:
    state = env.reset()

    while not done:
        next_state, reward, done, info = env.step(args.ways_be)

        for key, value in info.items():
            write_metrics(writer, key, value, env.steps)

    duration = env.get_experiment_duration()
    log.info("Experiment finished after {} steps.".format(env.steps))
    writer.add_hparams({'Action': args.ways_be},
                       {'Results/Violations Total': env.violations / env.steps, 'Results/Time': duration})

    writer.add_text('duration', form_duration(duration))
    writer.flush()

finally:
    writer.flush()
    writer.close()
    env.stop()
