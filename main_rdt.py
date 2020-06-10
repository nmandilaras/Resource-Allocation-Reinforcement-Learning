import time
from rdt_env import Rdt

cores_pid_hp, cores_pids_be = list(range(10)), list(range(10, 20))

env = Rdt(10, cores_pid_hp, cores_pids_be)

state = env.reset()

for i_episode in range(10):
    next_state, reward, done, _ = env.step(10)
    # do staff
    time.sleep(1)
