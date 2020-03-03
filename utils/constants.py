environment = 'CartPole-v1'

# 2 ή 4, 2, 6, 2
x_freq = 4
x_dot_freq = 2
theta_freq = 6
theta_dot_freq = 2

var_freq = [x_freq, x_dot_freq, theta_freq, theta_dot_freq]

train_episodes = 1000
eval_episodes = 200
max_steps = 195

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
