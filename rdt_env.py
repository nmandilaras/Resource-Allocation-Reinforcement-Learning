import gym
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding
from communication import get_latency
from pqos import Pqos
from pqos_handler import PqosHandlerCore, PqosHandlerPid
from random import randint


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, latency_thr, cores_pid_hp, cores_pids_be, num_ways=20, pqos_interface='MSR'):
        """ First of all we have to specify HP, BEs and assign them to different CPUs
            Need also to initialize RDT"""
        self.latency_thr = latency_thr
        self.num_ways = num_ways
        self.pqos_interface = pqos_interface
        # TODO launch hp and bes
        self.pqos = Pqos()
        self.pqos.init(pqos_interface)
        if pqos_interface == 'OS':
            self.pqos_handler = PqosHandlerPid(cores_pid_hp, cores_pids_be)
        else:
            self.pqos_handler = PqosHandlerCore(cores_pid_hp, cores_pids_be)
        self.pqos_handler.reset()
        self.pqos_handler.set_association_class()
        self.pqos_handler.print_association_config()

    def __reward_func(self, action_be_ways, hp_tail_latency):
        """Reward func """
        if hp_tail_latency >= self.latency_thr:
            reward = action_be_ways
            # NOTE by shaping the reward function in this way, we are making the assumption that progress of BEs is
            # depended by the LLC ways that are allocated to them at any point of their execution
        else:
            reward = - 2 * self.num_ways
        return reward

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that """
        # enforce the decision with PQOS
        self.pqos_handler.set_allocation_class(action_be_ways)
        self.pqos_handler.print_allocation_config()

        # start the stats record, the recorder will go to sleep and the it 'll send the results
        tail_latency = randint(0, 20)  # get_latency()  # NOTE this call will block
        sleep(0.5)

        self.pqos_handler.update()
        misses_be, socket_wide_bw = self.pqos_handler.get_hw_metrics()

        # other metrics? eg misbranched_ratio
        reward = self.__reward_func(action_be_ways, tail_latency)  # based on new metrics
        state = [tail_latency, misses_be, socket_wide_bw, action_be_ways]  # TODO form the state properly
        return state, reward, None, None

        # return next_state-new metrics, reward, done, info
        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning.
            We may also want to start from the beginning when one process finishes or all of them finishes
          """
        pass

    def render(self, **kwargs):
        pass

    def stop(self):
        self.pqos_handler.stop()
        self.pqos_handler.reset()
        self.pqos.fini()
