from enum import Enum

environment = 'CartPole-v0'

# 2 ή 4, 2, 6, 2
x_freq = 4
x_dot_freq = 2
theta_freq = 6
theta_dot_freq = 2

var_freq = [x_freq, x_dot_freq, theta_freq, theta_dot_freq]

TENSORBOARD = True

max_episodes = 500
max_steps = 195

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.0005  # drops to min around 100 episodes

EVAL_INTERVAL = 10
TERM_INTERVAL = 100 // EVAL_INTERVAL


class RLAlgorithms(Enum):
    Q_LEARNING = '0'
    SARSA = '1'
    DOUBLE_Q_LEARNING = '2'
