import gym
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """ First of all we have to specify HP, BEs and assign them to different CPUs
            Need also to initialize RDT"""
        pass

    def reward_func(self, metrics):
        "Reward func should be constituted by two components, one for "
        pass

    def step(self, hp_ways):
        """ At each step the agent specifies the number of ways that  """
        # enforce the decision with PQOS
        sleep(0.5)  # sleep around 500ms in order to observe the environment
        # use PQOS? to observe the new metrics
        # calculate reward based on new metrics
        ipc, mpkt, bw = None, None, None  # other metrics? eg misbranched_ration
        # return next_state-new metrics, reward, done, info
        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state
        pass

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning  """
        pass

    def render(self, **kwargs):
        pass
