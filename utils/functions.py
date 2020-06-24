import numpy as np
from utils import constants
import matplotlib.pyplot as plt
from utils.constants import LC_TAG
import re


def parse_num_list(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start), int(end)+1))

def plot_epsilon(epsilon):
    """ """
    plt.figure(2)
    plt.clf()
    plt.title('Epsilon...')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(range(len(epsilon)),epsilon)
    plt.pause(0.001)


def plot_rewards(episode_durations, eval_durations, completed=False, means=None):
    """ """
    figure = plt.figure(1)
    plt.clf()
    if completed:
        plt.title('Progress after {} training episodes'.format(len(episode_durations)))
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    train_i, train_duration = zip(*episode_durations.items())
    plt.plot(train_i, train_duration, label='train')
    try:
        eval_i, eval_duration = zip(*eval_durations.items())
        plt.plot(eval_i, eval_duration, marker=".", label='eval')
    except ValueError:
        pass
    plt.axhline(y=195, color='r')
    plt.legend(loc='best')
    plt.pause(0.001)  # pause a bit so that plots are updated

    return figure


def check_termination(eval_durations):
    return sum(list(eval_durations.values())[-constants.TERM_INTERVAL:]) / constants.TERM_INTERVAL >= 195


def write_metrics(tag, metrics, writer, step):
    ipc, misses, llc, mbl, mbr, latency = metrics
    header = '{}/'.format(tag)
    if tag == LC_TAG:
        writer.add_scalar(header + 'Latency', latency, step)

    writer.add_scalar(header + 'IPC', ipc, step)
    writer.add_scalar(header + 'Misses', misses, step)
    writer.add_scalar(header + 'LLC', llc, step)
    writer.add_scalar(header + 'MBL', mbl, step)
    writer.add_scalar(header + 'MBR', mbr, step)
    writer.flush()

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