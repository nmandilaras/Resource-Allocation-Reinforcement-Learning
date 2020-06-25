from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser, config_parser
from torch.utils.tensorboard import SummaryWriter
from utils.functions import write_metrics
from utils.constants import LC_TAG, BE_TAG

step = 0

parser = cmd_parser()
parser.add_argument('--warm-up', type=int, default=0, help='Time to collect metrics before/after bes execution')
parser.add_argument('--ways-be', type=int, default=-1, help='Ways to be allocated to best effort group')
args = parser.parse_args()
config_env, config_agent, config_misc = config_parser(args.config_file)

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

writer = SummaryWriter()

env = Rdt(config_env)

env.reset_pqos()
env.start_client()
log.debug("Mem client started. Warm up period follows.")

# collect tail latency and hw metrics before launching be
for i in range(args.warm_up):
    q95_latency = env.get_latency()
    env.update_hw_metrics()
    write_metrics(LC_TAG, env.get_lc_metrics(), writer, step)
    step += 1

env.start_bes()

log.info("Num of ways that are going to be statically allocated to BEs: {}".format(args.ways_be))

env.set_association_class(args.ways_be)
done = False
try:
    while not done:
        status = env.poll_bes()
        log.debug(status)
        done = all(status)

        q95_latency = env.get_latency()
        env.update_hw_metrics()
        reward = env.reward_func(args.ways_be, q95_latency)
        ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp = env.get_lc_metrics()
        ipc_be, misses_be, llc_be, mbl_be, mbr_be = env.get_be_metrics()

        write_metrics(LC_TAG, (ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp, q95_latency), writer, step)
        write_metrics(BE_TAG, (ipc_be, misses_be, llc_be, mbl_be, mbr_be, None), writer, step)

        # next_state, reward, done, info = env.step(args.ways_be)
        writer.add_scalar('Agent/Action', args.ways_be, step)
        writer.add_scalar('Agent/Reward', reward, step)

        step += 1

    log.info("Be finished")

finally:
    env.stop_bes()

    for i in range(args.warm_up):
        q95_latency = env.get_latency()
        env.update_hw_metrics()
        write_metrics(LC_TAG, env.get_lc_metrics(), writer, step)
        step += 1

    env.stop_client()
    env.stop_pqos()
