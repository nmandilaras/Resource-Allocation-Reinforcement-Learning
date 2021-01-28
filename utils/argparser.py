import argparse


def cmd_parser():
    """
    Parses command line arguments.

    Returns:
        an object with parsed command line arguments
    """

    description = 'RL Agent'
    parser = argparse.ArgumentParser(description=description, fromfile_prefix_chars='@')
    parser.add_argument('-i', '--interface', default='MSR', help='select pqos interface')
    parser.add_argument('-r', '--rps', type=int, default=10000, help='Requests per second that client should generate')
    parser.add_argument('-g', '--ratio', default='0.8', help='Ratio of get/set requests')
    parser.add_argument('-p', '--loader-dir', help='Path to memcached loader')
    parser.add_argument('-t', '--interval', default='200', help='Interval to wait after a decision in ms')
    parser.add_argument('--cores-lc', default="0", help='Cores in which lc critical service already run')
    parser.add_argument('--cores-be', default='1-9', help='Cores in which be process will be launched')
    parser.add_argument('--cores-client', default='10-13', help='Cores in which load client will be launched')
    parser.add_argument('--loader-threads', default='1', help='Number of workers for the load testing')
    parser.add_argument('--latency-thr', type=int, default=10, help='Q95 latency threshold in ms')
    parser.add_argument('--be-name', default='in-memory-small', help='Be name')
    parser.add_argument('--num-bes', type=int, default=1, help='Number of BE containers to be launched')
    parser.add_argument('--tensorboard', action='store_true', help='Enable Tensorboard')  # unused
    parser.add_argument('-c', '--config-file', default='configs/local', help='Path to config file')
    parser.add_argument('--comment', default='', help='Comment to add on tensorboard folder name as suffix')
    parser.add_argument('-q', '--quantile', default='.95', help='Choose quantile for which stats will be reported')
    parser.add_argument('-f', '--feature', default='MPKC', help='Hw feature to be used as input')
    parser.add_argument('-d', '--decay', default='0.0005', help='Epsilon decay rate')
    # parser.add_argument('--path-mem', help='')
    # nargs='+' all command-line args present are gathered into a list

    return parser
