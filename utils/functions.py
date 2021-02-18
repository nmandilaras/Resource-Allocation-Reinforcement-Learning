from utils.constants import metric_names
import re
import configparser


def config_parser(filename):
    config = configparser.ConfigParser()
    try:
        with open(filename) as f:
            config.read_file(f)
    except FileNotFoundError:
        raise FileNotFoundError("Filename {} does not exists. Exiting...".format(filename))

    return config


def parse_num_list(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start), int(end)+1))


def write_metrics(tboard_writer, tag, metrics, step):
    """ Used to write to Tensorboard environment related metrics. """
    header = '{}/'.format(tag)
    for metric, metric_name in zip(metrics, metric_names):
        if metric is not None:
            tboard_writer.add_scalar(header + metric_name, metric, step)
    tboard_writer.flush()


def form_duration(duration_minutes):
    """  """
    minutes = int(duration_minutes)
    seconds = int(round((duration_minutes % 1) * 60, 0))
    duration = str(minutes) + 'm' + str(seconds) + 's'

    return duration

# TODO remove this commented out code, check with jupyter notebook if similar code is present there
# use to log latency with this
# latency_per = np.percentile(latency_list, 99)
# latency_list_per = [min(i, latency_per) for i in latency_list]
# plt.plot(latency_list_per)
# plt.title('Effect of collocation in tail latency')
# plt.axvline(x=self.warm_up, color='g', linestyle='dashed', label='BEs starts')
# plt.axvline(x=len(latency_list_per) - self.warm_up, color='r', linestyle='dashed', label='BEs stops')
# plt.axhline(y=self.latency_thr, color='m', label='Latency threshold')
# plt.xlabel('Steps')
# plt.ylabel('Q95 Latency in ms')
# plt.legend(loc='best')
# plt.savefig('runs/collocation_{}.png'.format(datetime.today().strftime('%Y%m%d_%H%M%S')))
# plt.show()
