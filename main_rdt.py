from rdt_env import Rdt
import logging.config
from utils.argparser import init_parser

parser = init_parser()
args = parser.init_parser()

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

env = Rdt(args.latency_thr, args.cores_lc, args.cores_be, args.cores_client, args.path_mem, args.rps,
          args.client_threads, args.interval, args.be_name, args.ratio, args.num_bes, pqos_interface=args.interface)

num_of_observations = env.observation_space.shape[0]
high_intervals = env.observation_space.high
low_intervals = env.observation_space.low
num_of_actions = env.action_space.n

log.debug(high_intervals)
log.debug(low_intervals)
log.debug("Num of available actions: {}".format(num_of_actions))
log.debug("NUm of input features: {}".format(env.observation_space.shape[0]))

state = env.reset()
# next_state, reward, done, _ = env.step(10)

try:
   for i_episode in range(500000):
        next_state, reward, done, _ = env.step(10)
        log.debug('step')
        # do staff
        if done:
            log.info("Be finished")
            break
finally:
    env.stop()
