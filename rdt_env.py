import gym
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding
from communication import get_latency
from pqos_layer import PqosContextManager, MonitoringCore, MonitoringPid


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, latency_thr, cores_pids, events, num_ways=20, pqos_interface='MSR'):
        """ First of all we have to specify HP, BEs and assign them to different CPUs
            Need also to initialize RDT"""
        self.latency_thr = latency_thr
        self.num_ways = num_ways
        self.pqos_interface = pqos_interface
        with PqosContextManager(pqos_interface):
            if pqos_interface == 'OS':
                self.monitoring = MonitoringPid(cores_pids, events)
            else:
                self.monitoring = MonitoringCore(cores_pids, events)
            self.monitoring.setup()

    def cleanup(self):
        with PqosContextManager(self.pqos_interface):
            self.monitoring.stop()

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
        """ At each step the agent specifies the number of ways that  """
        # enforce the decision with PQOS
        with PqosContextManager(self.pqos_interface):
            pass

        # start the stats record, the recorder will go to sleep and the it 'll send the results
        tail_latency = get_latency()  # NOTE this call will block

        with PqosContextManager(self.pqos_interface):
            self.monitoring.update()
            self.monitoring.print_data()
        ipc, mpkt, bw = None, None, None  # other metrics? eg misbranched_ratio
        reward = self.__reward_func(action_be_ways, tail_latency)  # based on new metrics
        state = None
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
