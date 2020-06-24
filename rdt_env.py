import gym
from time import sleep
from gym import spaces
from communication import get_latency
from pqos import Pqos
from pqos_handler import PqosHandlerCore, PqosHandlerPid, PqosHandlerMock
from utils.functions import parse_num_list
import numpy as np
import subprocess
import docker
import logging.config
from utils.config_constants import *

logging.config.fileConfig('logging.conf')  # TODO path!
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
        self.action_interval = config_env[ACTION_INTERVAL]
        self.pqos_interface = config_env[PQOS_INTERFACE]
        self.be_name = config_env[BE_NAME]
        self.ratio = config_env[GET_SET_RATIO]
        self.num_bes = int(config_env[NUM_BES])
        self.container_bes = []
        cores_pid_hp_range = parse_num_list(config_env[CORES_LC])
        self.cores_pids_be_range = parse_num_list(self.cores_pids_be)
        # log.debug(cores_pid_hp_range)
        # log.debug(self.cores_pids_be_range)
        # log.debug(cores_loader)

        self.action_space = spaces.Discrete(int(config_env[NUM_WAYS]))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([20, 3 * 1e7, 6 * 1e3, self.action_space.n-1], dtype=np.float32),
            dtype=np.float32)

        self.mem_client = None
        self.container_be = None

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
                                            dataset, '-s', servers, '-g', self.ratio, '-c', '200', '-e', '-w',
                                            self.loader_threads, '-T', self.action_interval, '-r', str(self.rps)])
        sleep(10)  # wait in order to bind the socket

    def stop_client(self):
        self.mem_client.terminate()
        sleep(0.5)
        while self.mem_client.poll() is None:
            log.debug("Unable to shutdown mem client. Retrying...")
            self.mem_client.terminate()

    def start_bes(self):
        """ Check if bes are already initialized and restarts them otherwise they will be launched"""
        log.debug("BE {} to be started.".format(self.be_name))
        client = docker.from_env()
        if not self.container_bes:
            container, command, volume = bes[self.be_name]
            cores_per_be = int(len(self.cores_pids_be_range) / self.num_bes)
            for i in range(self.num_bes):
                cpuset = ','.join(map(str, self.cores_pids_be_range[i * cores_per_be: (i + 1) * cores_per_be]))
                log.debug("Cores for be: {}".format(cpuset))
                container_be = client.containers.run(container,
                                                      command=command, name='be_' + str(i),
                                                      cpuset_cpus=cpuset, volumes_from=[volume], detach=True)
                self.container_bes.append(container_be)
        else:
            for container_be in self.container_bes:
                container_be.restart()

    def poll_bes(self):
        status = []
        for container_be in self.container_bes:
            container_be.reload()
            if container_be.status == 'exited':
                status.append(True)
            else:
                status.append(False)

        return status

    def stop_bes(self):
        for container_be in self.container_bes:
            container_be.stop()
            container_be.remove()

    @staticmethod
    def get_latency():
        return get_latency()

    def update_hw_metrics(self):
        self.pqos_handler.update()

    def get_lc_metrics(self):  # to be called after update metrics
        return self.pqos_handler.get_hp_metrics()

    def get_be_metrics(self):
        return self.pqos_handler.get_be_metrics()

    def __get_next_state(self, action_be_ways):
        # poll metrics so the next poll will contains deltas from this point just after the action
        self.pqos_handler.update()
        # start the stats record, the recorder will go to sleep and the it 'll send the results
        tail_latency = get_latency()  # NOTE this call will block

        self.pqos_handler.update()
        ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp = self.pqos_handler.get_hp_metrics()
        ipc_be, misses_be, llc_be, mbl_be, mbr_be = self.pqos_handler.get_be_metrics()
        socket_wide_bw = mbl_hp + mbl_be
        info = {'Latency Critical': (ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp, tail_latency),
                'Best Effort': (ipc_be, misses_be, llc_be, mbl_be, mbr_be, None)}

        state = [tail_latency, misses_be, socket_wide_bw, action_be_ways]

        # normalize the state
        state_normalized = [min(metric / max_val, 1) for metric, max_val in zip(state, self.observation_space.high)]

        return state_normalized, info, tail_latency

    def __reward_func(self, action_be_ways, hp_tail_latency):
        """Reward func """
        if hp_tail_latency < self.latency_thr:
            reward = action_be_ways
            # NOTE by shaping the reward function in this way, we are making the assumption that progress of BEs is
            # depended by the LLC ways that are allocated to them at any point of their execution
        else:
            reward = - 2 * self.action_space.n
        return reward

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning.
            We may also want to start from the beginning when one process finishes or all of them finishes
          """
        self.reset_pqos()

        # (re)launch the load tester
        if self.mem_client is not None:
            self.stop_client()
        self.start_client()
        log.debug("Mem client started. Warm up period follows.")

        # (re)launch the be containers
        self.start_bes()
        log.debug('BEs started')

        state, _, _ = self.__get_next_state(self.action_space.n)  # TODO check this, with how many ways be starts?

        return state

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that are assigned to the be"""

        log.debug("Action selected: {}".format(action_be_ways))
        status = self.poll_bes()
        log.debug(status)
        done = all(status)

        err_msg = "%r (%s) invalid" % (action_be_ways, type(action_be_ways))
        assert self.action_space.contains(action_be_ways), err_msg

        # enforce the decision with PQOS
        self.pqos_handler.set_allocation_class(action_be_ways)
        self.pqos_handler.print_allocation_config()

        state, info, tail_latency = self.__get_next_state(action_be_ways)

        reward = self.__reward_func(action_be_ways, tail_latency)  # based on new metrics

        return state, reward, done, info

        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state

    def render(self, **kwargs):
        pass

    def stop(self):
        log.debug('Stopping everything!')

        # stop and remove the be containers
        self.stop_bes()

        # stop the mem client
        self.stop_client()

        # stop pqos
        self.stop_pqos()
