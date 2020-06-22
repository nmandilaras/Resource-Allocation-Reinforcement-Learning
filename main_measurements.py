from rdt_env import Rdt
import logging.config
from utils.argparser import init_parser
from torch.utils.tensorboard import SummaryWriter

step = 0


def write_metrics(tag, metrics, latency=None):
    ipc, misses, llc, mbl, mbr = metrics
    if tag == 'Latency_Critical':
        writer.add_scalar('Metrics/Latency_Critical/Latency', latency, step)
    header = 'Metrics/{}/'.format(tag)
    writer.add_scalar(header + 'IPC', ipc, step)
    writer.add_scalar(header + 'Misses', misses, step)
    writer.add_scalar(header + 'LLC', llc, step)
    writer.add_scalar(header + 'MBL', mbl, step)
    writer.add_scalar(header + 'MBR', mbr, step)


parser = init_parser()
parser.add_argument('--warm-up', type=int, default=30, help='Time to collect metrics, in seconds')
# parser.add_argument('--path-mem', help='')
args = parser.parse_args()

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

writer = SummaryWriter()

env = Rdt(args.latency_thr, args.cores_lc, args.cores_be, args.cores_client, args.path_mem, args.rps,
          args.client_threads, args.interval, args.be_name, args.ratio, args.num_bes, pqos_interface=args.interface)

env.reset_pqos()
env.start_client()
log.debug("Mem client started. Warm up period follows.")

# collect tail latency and hw metrics before launching be
for i in range(args.warm_up):
    q95_latency = env.get_latency()
    env.update_hw_metrics()
    write_metrics('Latency_Critical', env.get_lc_metrics(), q95_latency)
    step += 1

env.start_bes()

done = False
try:
    while not done:
        status = env.poll_bes()
        log.debug(status)
        done = all(status)

        q95_latency = env.get_latency()
        env.update_hw_metrics()
        write_metrics('Latency_Critical', env.get_lc_metrics(), q95_latency)
        write_metrics('Best_Effort', env.get_lc_metrics())
        step += 1

    log.info("Be finished")

finally:
    env.stop_bes()

    for i in range(args.warm_up):
        q95_latency = env.get_latency()
        env.update_hw_metrics()
        write_metrics('Latency_Critical', env.get_lc_metrics(), q95_latency)
        step += 1

    env.stop_client()
    env.stop_pqos()
