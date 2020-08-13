import gym
import ast
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

file = open("docker_containers", "r")
contents = file.read()
bes = ast.literal_eval(contents)


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
        self.ratio = config_env[GET_SET_RATIO]
        self.exponential_dist = config_env[EXP_DIST]
        self.num_total_bes = int(config_env[NUM_BES])
        self.container_bes = []
        cores_pid_hp_range = parse_num_list(config_env[CORES_LC])
        self.cores_pids_be_range = parse_num_list(self.cores_pids_be)
        self.cores_per_be = int(config_env[CORES_PER_BE])
        self.violations = 0  # calculate violations
        self.steps = 1
        self.start_time_bes = None
        self.stop_time_bes = None
        self.interval_bes = None  # in minutes
        self.seed = int(config_env[SEED])
        self.penalty_coef = float(config_env[PEN_COEF])
        self.client = docker.from_env()
        self.issued_bes = 0
        self.finished_bes = 0
        self.generator = None
        self.cores_map = lambda i: ','.join(map(str, self.cores_pids_be_range[i * self.cores_per_be: (i + 1) * self.cores_per_be]))
        self.be_name = ast.literal_eval(config_env[BE_NAME]) if config_env[BE_NAME] else None
        self.be_repeated = int(config_env[BE_REPEATED])
        self.be_quota = self.be_repeated
        self.last_be = None
        self.new_be = False

        self.action_space = spaces.Discrete(int(config_env[NUM_WAYS]))
        # latency, mpki_be # used to be 2*1e6, 5*1e7, ways_be # 14 me 30 gia mpc kai be=mcf
        # for gradient boost high in misses raised to 20 from 14
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([14, self.action_space.n-1], dtype=np.float32),
            dtype=np.float32)

        # # latency, ipc 0.82-0.87, ways_be
        # self.observation_space = spaces.Box(
        #     low=np.array([5, 0]), high=np.array([15, self.action_space.n-1], dtype=np.float32),
        #     dtype=np.float32)

        # # latency, ipc_lc, mpki_lc, bw_lc, ways_be
        # self.observation_space = spaces.Box(
        #     low=np.array([5, 0.75, 4, 200, 0]), high=np.array([15, 0.9, 5, 1000, self.action_space.n-1], dtype=np.float32),
        #     dtype=np.float32)

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
        self.mem_client = subprocess.Popen(['taskset', '--cpu-list', self.cores_loader, loader, '-a', dataset,
                                            '-s', servers, '-g', self.ratio, '-c', self.loader_conn, '-w',
                                            self.loader_threads, '-T', self.action_interval, '-r', str(self.rps),
                                            self.exponential_dist])
        sleep(10)  # wait in order to bind the socket

    def stop_client(self):
        self.mem_client.terminate()
        sleep(0.5)
        while self.mem_client.poll() is None:
            log.debug("Unable to shutdown mem client. Retrying...")
            self.mem_client.terminate()

    def _select_be(self):

        if self.be_quota == self.be_repeated:
            log.info("Quota expired new be will be issued!")
            self.be_quota = 1
            self.last_be = self.be_name.pop(0) if self.be_name else self.generator.choice(list(bes.keys()))
            self.new_be = True
            # increase exploration
            # erase memory
        else:
            self.be_quota += 1

        return self.last_be

    def _start_be(self, cores):
        """ Start a container on specified cores """

        log.info('New BE will be issued on core(s): {} at step: {}'.format(cores, self.steps))

        be = self._select_be()
        log.info('Selected Job: {}'.format(be))
        container, command, volume = bes[be]
        container_be = self.client.containers.run(container, command=command, name='be_' + cores.replace(",", "_"),
                                                  cpuset_cpus=cores, volumes_from=[volume] if volume is not None else [],
                                                  detach=True)
        self.issued_bes += 1

        return container_be

    def start_bes(self):
        """ Launches bes """

        num_startup_bes = min(len(self.cores_pids_be_range) // self.cores_per_be, self.num_total_bes)
        self.container_bes = [self._start_be(self.cores_map(i)) for i in range(num_startup_bes)]

        self.start_time_bes = time.time()

    def poll_bes(self):
        status = []
        for container_be in self.container_bes:
            container_be.reload()
            if container_be.status == 'exited':
                self._stop_be(container_be)
                status.append(True)
                self.finished_bes += 1
            else:
                status.append(False)

        # issue new dockers on cores that finished execution
        for i, status in enumerate(status):
            if status:
                self.container_bes[i] = self._start_be(self.cores_map(i))
                log.info("Finished Bes: {}/{}".format(self.finished_bes, self.num_total_bes))

        done = self.finished_bes >= self.num_total_bes

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

        bw_socket_wide = mbl_hp_ps + mbl_be_ps
        bw_lc = mbl_hp_ps + mbr_hp_ps
        # misses_be = misses_be / (int(self.action_interval) // 50)
        info = {LC_TAG: (ipc_hp, misses_hp, llc_hp, mbl_hp_ps, mbr_hp_ps, tail_latency, rps),
                BE_TAG: (ipc_be, misses_be, llc_be, mbl_be_ps, mbr_be_ps, None, None)}

        state = [misses_be, action_be_ways]

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
            reward = - self.penalty_coef * self.action_space.n
            self.violations += 1
        return reward

    def reset(self):
        """ Probably when we end up in a very bad situation we want to start from the beginning.
            We may also want to start from the beginning when one process finishes or all of them finishes
          """
        # random.seed(self.seed)
        self.generator = random.Random(self.seed)

        self.reset_pqos()

        # (re)launch the load tester
        if self.mem_client is not None:
            self.stop_client()
        self.start_client()
        log.debug("Mem client started.")

        # launch the be containers
        self.stop_bes()
        self.start_bes()
        log.debug('BEs started')

        state, _, _ = self.__get_next_state(self.action_space.n)  # we start with both groups sharing all ways

        return state

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that are assigned to the be"""

        # log.debug("Action selected: {}".format(action_be_ways))
        self.new_be = False
        # check once for every 1000ms
        done = self.determine_termination() if self.steps % (1000 // int(self.action_interval)) == 0 else False

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

        self.steps += 1

        return state, reward, done, info, self.new_be

        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state

    def render(self, **kwargs):
        pass

    def stop(self):
        log.warning('Stopping everything!')

        log.info('Percentage of violations: {}'.format(self.violations / self.steps))

        # stop and remove the be containers
        self.stop_bes()

        # stop the mem client
        self.stop_client()

        # stop pqos
        self.stop_pqos()
