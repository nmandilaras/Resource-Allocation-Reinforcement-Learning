import gym
import time
from rdt_env import Rdt

env = Rdt(10)

state = env.reset()

for i_episode in range(10):
    next_state, reward, done, _ = env.step(10)
    # do staff
    time.sleep(1)
