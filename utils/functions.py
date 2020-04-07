import numpy as np
from utils import constants
import matplotlib.pyplot as plt


def plot_epsilon(epsilon):
    """ """
    plt.figure(2)
    plt.clf()
    plt.title('Epsilon...')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(range(len(epsilon)),epsilon)
    plt.pause(0.001)

def plot_durations(episode_durations, eval_durations, completed=False, means=None):
    """ For cartpole since reward is 1 for each step we are upright this plot represents rewards also"""
    figure = plt.figure(1)
    plt.clf()
    if completed:
        plt.title('Progress after {} training episodes'.format(len(episode_durations)))
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    train_i, train_duration = zip(*episode_durations.items())
    plt.plot(train_i, train_duration, label='train')
    try:
        eval_i, eval_duration = zip(*eval_durations.items())
        plt.plot(eval_i, eval_duration, marker=".", label='eval')
    except ValueError:
        pass
    plt.axhline(y=195, color='r')
    plt.legend(loc='best')
    # Take 100 episode averages and plot them too
    # if len(episode_durations) >= 100:
    #     mean = np.mean(episode_durations[-100:])
    #     means.append(mean)
    #     plt.plot(means)
    # else:
    #     means.append(0)
    plt.pause(0.001)  # pause a bit so that plots are updated

    return figure


def check_termination(eval_durations):
    return sum(list(eval_durations.values())[-constants.TERM_INTERVAL:]) / constants.TERM_INTERVAL >= 195
