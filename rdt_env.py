import gym
from time import sleep
from gym import spaces
from communication import get_loader_stats
from pqos import Pqos
from pqos_handler import PqosHandlerCore, PqosHandlerPid, PqosHandlerMock
from utils.functions import parse_num_list
import numpy as np
import subprocess
import docker
import logging.config
from utils.constants import LC_TAG, BE_TAG
from utils.config_constants import *
import time
import random

logging.config.fileConfig('logging.conf')  # NOTE path!
log = logging.getLogger('simpleExample')

bes = {  # NOTE better to get those from file
    'in-memory': ('zilutian/in-memory-analytics:amd64', '/data/ml-latest /data/myratings.csv --driver-memory 6g --executor-memory 16g', 'data'),
    'in-memory-small': ('zilutian/in-memory-analytics:amd64', '/data/ml-latest-small /data/myratings.csv', 'data'),
    'graphs': ('cloudsuite/graph-analytics', '--driver-memory 6g --executor-memory 16g', 'data-twitter')
}


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_env):
        """ """
        self.latency_thr = int(config_env[LATENCY_thr])
        self.cores_pids_be = config_env[CORES_BE]
        self.cores_loader = config_env[CORES_LOADER]
        self.loader_dir = config_env[LOADER_DIR]
        self.rps = int(config_env[LOADER_RPS])
        self.loader_threads = config_env[LOADER_THREADS]
        self.loader_conn = config_env[LOADER_CONN]
        self.action_interval = config_env[ACTION_INTERVAL]
        self.pqos_interface = config_env[PQOS_INTERFACE]
        # self.be_name = config_env[BE_NAME]
        self.ratio = config_env[GET_SET_RATIO]
        self.num_total_bes = int(config_env[NUM_BES])
        self.container_bes = []
        cores_pid_hp_range = parse_num_list(config_env[CORES_LC])
        self.cores_pids_be_range = parse_num_list(self.cores_pids_be)
        self.cores_per_be = 1  # NOTE discontinued
        self.violations = 0  # calculate violations
        self.start_time_bes = None
        self.stop_time_bes = None
        self.interval_bes = None  # in minutes
        self.seed = int(config_env[SEED])
        self.client = docker.from_env()
        self.issued_bes = 0

        self.action_space = spaces.Discrete(int(config_env[NUM_WAYS]))
        # # latency, misses, bw, ways_be
        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0]), high=np.array([20, 10, 1e5, self.action_space.n-1], dtype=np.float32),
        #     dtype=np.float32)

        # latency, ipc, ways_be
        self.observation_space = spaces.Box(
            low=np.array([5, 0.7, 0]), high=np.array([15, 0.9, self.action_space.n-1], dtype=np.float32),
            dtype=np.float32)

        self.mem_client = None
        self.container_be = None
        self.previous_action = -1  # -1 action means all ways available to all groups

        # initialize pqos
        if self.pqos_interface == 'none':
            self.pqos_handler = PqosHandlerMock()
        else:
            self.pqos = Pqos()
            self.pqos.init(self.pqos_interface)
            if self.pqos_interface == 'OS':
                self.pqos_handler = PqosHandlerPid(cores_pid_hp_range, self.cores_pids_be_range)
            else:
                self.pqos_handler = PqosHandlerCore(cores_pid_hp_range, self.cores_pids_be_range)

    def reset_pqos(self):
        self.pqos_handler.reset()
        self.pqos_handler.setup()
        self.pqos_handler.set_association_class()
        self.pqos_handler.print_association_config()
        self.previous_action = -1

    def stop_pqos(self):
        self.pqos_handler.stop()
        self.pqos_handler.reset()
        if self.pqos_interface != 'none':
            self.pqos.fini()

    def start_client(self):
        """  """
        loader = '{}/loader'.format(self.loader_dir)
        dataset = '{}/twitter_dataset/twitter_dataset_30x'.format(self.loader_dir)
        servers = '{}/docker_servers.txt'.format(self.loader_dir)
        self.mem_client = subprocess.Popen(['taskset', '--cpu-list', self.cores_loader, loader, '-a',
                                            dataset, '-s', servers, '-g', self.ratio, '-c', self.loader_conn, '-w',
                                            self.loader_threads, '-T', self.action_interval, '-r', str(self.rps)])
        sleep(10)  # wait in order to bind the socket

    def stop_client(self):
        self.mem_client.terminate()
        sleep(0.5)
        while self.mem_client.poll() is None:
            log.debug("Unable to shutdown mem client. Retrying...")
            self.mem_client.terminate()

    def _start_be(self, core):
        """ Start a container on specified core """

        log.info('New BE will be issued on core: {}'.format(core))

        container, command, volume = random.choice(list(bes.values()))

        container_be = self.client.containers.run(container, command=command, name='be_' + core,
                                                  cpuset_cpus=core, volumes_from=[volume], detach=True)
        self.issued_bes += 1

        return container_be

    def start_bes(self):
        """ Check if bes are already initialized and restarts them otherwise they will be launched"""

        num_startup_bes = min(len(self.cores_pids_be_range), self.num_total_bes)
        self.container_bes = [self._start_be(str(self.cores_pids_be_range[i])) for i in range(num_startup_bes)]

        self.start_time_bes = time.time()

    def poll_bes(self):
        status = []
        for container_be in self.container_bes:
            container_be.reload()
            if container_be.status == 'exited':
                self._stop_be(container_be)
                status.append(True)
            else:
                status.append(False)

        # issue new dockers on cores that finished execution
        for i, status in enumerate(status):
            if status:
                self.container_bes[i] = self._start_be(str(self.cores_pids_be_range[i]))
                # TODO consider the possibility to increase exploration

        done = self.issued_bes > self.num_total_bes

        return done

    @staticmethod
    def _stop_be(container_be):
        """"""
        container_be.stop()
        container_be.remove()

    def stop_bes(self):
        for container_be in self.container_bes:
            self._stop_be(container_be)

    def determine_termination(self):
        done = self.poll_bes()
        # log.debug(status)
        # done = any(status)
        if done:
            self.stop_time_bes = time.time()
            self.interval_bes = (self.stop_time_bes - self.start_time_bes) / 60

        return done

    @staticmethod
    def get_loader_stats():
        return get_loader_stats()

    def update_hw_metrics(self):
        self.pqos_handler.update()

    def get_lc_metrics(self, time_interval):  # to be called after update metrics
        return self.pqos_handler.get_hp_metrics(time_interval)

    def get_be_metrics(self, time_interval):
        return self.pqos_handler.get_be_metrics(time_interval)

    def set_association_class(self, action_be_ways):
        self.pqos_handler.set_allocation_class(action_be_ways)

    @staticmethod
    def _normalize(metric, min_val, max_val):
        if metric > max_val:
            return 1.0
        elif metric < min_val:
            return 0.0
        else:
            return (metric - min_val) / (max_val - min_val)

    def __get_next_state(self, action_be_ways):
        # poll metrics so the next poll will contains deltas from this point just after the action
        self.pqos_handler.update()
        start_time = time.time()
        # start the stats record, the recorder will go to sleep and the it 'll send the results
        tail_latency, rps = self.get_loader_stats()  # NOTE this call will block

        self.pqos_handler.update()
        time_interval = time.time() - start_time
        ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps = self.pqos_handler.get_hp_metrics(time_interval)
        ipc_be, misses_be, llc_be, mbl_be_ps, mbr_be_ps = self.pqos_handler.get_be_metrics(time_interval)

        socket_wide_bw = mbl_hp_ps + mbl_be_ps
        info = {LC_TAG: (ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps, tail_latency, rps),
                BE_TAG: (ipc_be, misses_be, llc_be, mbl_be_ps, mbr_be_ps, None, None)}

        state = [tail_latency, ipc_hp, action_be_ways]

        # normalize the state
        state_normalized = [self._normalize(metric, min_val, max_val) for metric, min_val, max_val in
                            zip(state, self.observation_space.low, self.observation_space.high)]

        return state_normalized, info, tail_latency

    def reward_func(self, action_be_ways, hp_tail_latency):
        """Reward func """
        if hp_tail_latency < self.latency_thr:
            reward = action_be_ways
            # NOTE by shaping the reward function in this way, we are making the assumption that progress of BEs is
            # depended by the LLC ways that are allocated to them at any point of their execution
        else:
            reward = - 2 * self.action_space.n
            self.violations += 1
        return reward

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning.
            We may also want to start from the beginning when one process finishes or all of them finishes
          """
        random.seed(self.seed)

        self.reset_pqos()

        # (re)launch the load tester
        if self.mem_client is not None:
            self.stop_client()
        self.start_client()
        log.debug("Mem client started. Warm up period follows.")

        # launch the be containers
        self.stop_bes()
        self.start_bes()
        log.debug('BEs started')

        state, _, _ = self.__get_next_state(self.action_space.n)  # we start with both groups sharing all ways

        return state

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that are assigned to the be"""

        log.debug("Action selected: {}".format(action_be_ways))
        done = self.determine_termination()

        # err_msg = "%r (%s) invalid" % (action_be_ways, type(action_be_ways))
        # assert self.action_space.contains(action_be_ways), err_msg

        # Does this check cause any problem ?
        if action_be_ways != self.previous_action:  # avoid enforcing decision when nothing changes
            # enforce the decision with PQOS
            self.pqos_handler.set_allocation_class(action_be_ways)
            # self.pqos_handler.print_allocation_config()
            self.previous_action = action_be_ways

        state, info, tail_latency = self.__get_next_state(action_be_ways)

        reward = self.reward_func(action_be_ways, tail_latency)  # based on new metrics

        return state, reward, done, info

        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state

    def render(self, **kwargs):
        pass

    def stop(self):
        log.warning('Stopping everything!')

        # stop and remove the be containers
        self.stop_bes()

        # stop the mem client
        self.stop_client()

        # stop pqos
        self.stop_pqos()
