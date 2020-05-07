import logging.config
from archive.tic_tac_toe_env import TicTac4

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('simpleExample')
env = TicTac4()

env.reset()
print(env.step(5))
print(env.step(3))
# env.render()