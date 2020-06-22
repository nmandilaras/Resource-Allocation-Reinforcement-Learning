import argparse


def init_parser():
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
    parser.add_argument('--tensorboard', action='store_true', help='Enable Tensorboard')  # unused
    # parser.add_argument('--path-mem', help='')
    # nargs='+' all command-line args present are gathered into a list

    return parser
