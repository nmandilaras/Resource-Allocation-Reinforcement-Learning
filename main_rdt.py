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
    parser.add_argument('-r', '--rps', help='Requests per second that memcached client should generate')
    parser.add_argument('-p', '--path-mem', help='Path to memcached loader')
    # nargs='+' all command-line args present are gathered into a list

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    cores_pid_hp, cores_pids_be = list(range(5)), list(range(5, 10))

    logging.config.fileConfig('logging.conf')
    log = logging.getLogger('simpleExample')

    latency_thr = 10  # q95 in ms, based on cloudsuite's documentation
    env = Rdt(latency_thr, cores_pid_hp, cores_pids_be, rps=args.rps, pqos_interface=args.interface)

    num_of_observations = env.observation_space.shape[0]
    high_intervals = env.observation_space.high
    low_intervals = env.observation_space.low
    num_of_actions = env.action_space.n

    log.debug(high_intervals)
    log.debug(low_intervals)
    log.debug(num_of_actions)
    log.debug(env.observation_space.shape[0])

    state = env.reset()

    try:
        for i_episode in range(5):
            next_state, reward, done, _ = env.step(10)

            # do staff
            time.sleep(1)
    finally:
        env.stop()
