import numpy as np
import matplotlib.pyplot as plt


def plot_durations(episode_durations, means, eval_durations):
    """ For cartpole since reward is 1 for each step we are upright this plot represents rewards also"""
    plt.figure(2)
    plt.clf()
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
