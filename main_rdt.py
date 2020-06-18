import time
from rdt_env import Rdt
import logging.config
import argparse


def parse_args():
    """
    Parses command line arguments.

    Returns:
        an object with parsed command line arguments
    """

    description = 'RL Agent'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--interface', default='MSR', help='select pqos interface')
    parser.add_argument('-r', '--rps', type=int, default=20000, help='Requests per second that client should generate')
    parser.add_argument('-g', '--ratio', default='0.8', help='Ratio of get/set requests')
    parser.add_argument('-p', '--path-mem', help='Path to memcached loader')
    parser.add_argument('-t', '--interval', default='200', help='Interval to wait after a decision in ms')
    parser.add_argument('--cores-lc', default="0-3", help='Cores in which lc critical service already run')
    parser.add_argument('--cores-be', default='4-9', help='Cores in which be process will be launched')
    parser.add_argument('--cores-client', default='10-14', help='Cores in which load client will be launched')
    parser.add_argument('--client-threads', default='1', help='Number of clients for the load testing')
    parser.add_argument('--latency-thr', type=int, default=10, help='Q95 latency threshold in ms')
    parser.add_argument('--be-name', default='in-memory-small', help='Be name')
    parser.add_argument('--num-bes', type=int, default=1, help='Number of BE containers to be launched')
    # parser.add_argument('--path-mem', help='')
    # nargs='+' all command-line args present are gathered into a list

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

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
