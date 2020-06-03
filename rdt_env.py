import gym
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, hp_thr, num_ways=20):
        """ First of all we have to specify HP, BEs and assign them to different CPUs
            Need also to initialize RDT"""
        self.hp_thr = hp_thr
        self.num_ways = num_ways

    def reward_func(self, action_be_ways, hp_perf):
        """Reward func """
        if hp_perf >= self.hp_thr:
            reward = action_be_ways
            # be shaping the reward function in this way, we are making the assumption that progress of BEs is depended
            # by the LLC ways that are allocated to them at any point of their execution
        else:
            reward = - 2 * self.num_ways
        return reward

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that  """
        # enforce the decision with PQOS
        # start the stats record
        sleep(0.5)  # sleep around 500ms in order to observe the environment
        # stop the recording and collect results
        # use PQOS? to observe the new metrics
        ipc, mpkt, bw = None, None, None  # other metrics? eg misbranched_ration
        # tail_latency = 
        # calculate reward based on new metrics
        # return next_state-new metrics, reward, done, info
        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state
        pass

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning.
            We may also want to start from the beginning when one process finishes or all of them finishes
          """
        pass

    def render(self, **kwargs):
        pass
