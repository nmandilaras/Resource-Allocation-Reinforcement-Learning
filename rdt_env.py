import gym
from time import sleep
from gym import error, spaces, utils
from gym.utils import seeding
from communication import get_latency
from pqos import Pqos
from pqos_handler import PqosHandlerCore, PqosHandlerPid, PqosHandlerMock
from utils.functions import parse_num_list
import numpy as np
import subprocess
import docker
import logging.config
import matplotlib.pyplot as plt
from datetime import datetime

WARM_UP_PERIOD = 30

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')
latency_list = []

bes = {
    'in-memory': ('zilutian/in-memory-analytics:amd64', '/data/ml-latest /data/myratings.csv --driver-memory 6g --executor-memory 16g', 'data'),
    'in-memory-small': ('zilutian/in-memory-analytics:amd64', '/data/ml-latest-small /data/myratings.csv', 'data'),
    'graphs': ('cloudsuite/graph-analytics', '--driver-memory 6g --executor-memory 16g', 'data-twitter')
}


class Rdt(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, latency_thr, cores_pid_hp, cores_pids_be, cores_client, path_mem, rps, clnt_thrds, wait_interval,
                 be_name, num_ways=20, pqos_interface='MSR'):
        """ First of all we have to specify HP, BEs and assign them to different CPUs
            Need also to initialize RDT"""
        self.latency_thr = latency_thr
        self.cores_pids_be = cores_pids_be
        self.cores_client = cores_client
        self.path_mem = path_mem
        self.rps = rps
        self.clnt_thrds = clnt_thrds
        self.wait_interval = wait_interval
        self.pqos_interface = pqos_interface
        self.warm_up = WARM_UP_PERIOD * 1000 / int(self.wait_interval)
        self.be_name = be_name
        cores_pid_hp_range = parse_num_list(cores_pid_hp)
        cores_pids_be_range = parse_num_list(cores_pids_be)
        # log.debug(cores_pid_hp_range)
        # log.debug(cores_pids_be_range)
        # log.debug(cores_client)

        self.action_space = spaces.Discrete(num_ways)  # TODO maybe will reduce this, only few ways to the be
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 1]), high=np.array([np.finfo(np.float32).max, np.finfo(np.float32).max,
                                        np.finfo(np.float32).max, self.action_space.n], dtype=np.float32),
            dtype=np.float32)

        self.mem_client = None
        self.container_be = None

        # initialize pqos
        if pqos_interface == 'none':
            self.pqos_handler = PqosHandlerMock()
        else:
            self.pqos = Pqos()
            self.pqos.init(pqos_interface)
            if pqos_interface == 'OS':
                self.pqos_handler = PqosHandlerPid(cores_pid_hp_range, cores_pids_be_range)
            else:
                self.pqos_handler = PqosHandlerCore(cores_pid_hp_range, cores_pids_be_range)

    def __start_client(self):
        """  """
        loader = '{}/loader'.format(self.path_mem)
        dataset = '{}/twitter_dataset/twitter_dataset_3x'.format(self.path_mem)
        servers = '{}/docker_servers.txt'.format(self.path_mem)
        self.mem_client = subprocess.Popen(['taskset', '--cpu-list', str(self.cores_client), loader, '-a',
                                            dataset, '-s', servers, '-g', '0.8', '-c', '200', '-e', '-w',
                                            self.clnt_thrds, '-T', self.wait_interval, '-r', str(self.rps)])
        sleep(10)  # wait in order to bind the socket

    def __stop_client(self):
        self.mem_client.terminate()
        sleep(0.5)
        while self.mem_client.poll() is None:
            log.debug("Unable to shutdown mem client. Retrying...")
            self.mem_client.terminate()

    def __start_be(self):
        """ """
        log.debug("BE {} to be started".format(self.be_name))
        client = docker.from_env()
        container, command, volume = bes[self.be_name]
        self.container_be = client.containers.run(container,
                                                  command=command, name='be',
                                                  cpuset_cpus=self.cores_pids_be, volumes_from=[volume], detach=True)

    def __stop_be(self):
        try:
            self.container_be.reload()
            self.container_be.stop()
            self.container_be.remove()
        except:
            pass

    def __get_next_state(self, action_be_ways):
        # poll metrics so the next poll will contains deltas from this point just after the action
        self.pqos_handler.update()
        # start the stats record, the recorder will go to sleep and the it 'll send the results
        tail_latency = get_latency()  # randint(0, 20)  # NOTE this call will block
        latency_list.append(tail_latency)

        self.pqos_handler.update()
        misses_be, socket_wide_bw = self.pqos_handler.get_hw_metrics()

        state = [tail_latency, misses_be, socket_wide_bw, action_be_ways]  # TODO form the state properly
        return state

    def __reward_func(self, action_be_ways, hp_tail_latency):
        """Reward func """
        if hp_tail_latency >= self.latency_thr:
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
        self.pqos_handler.reset()
        self.pqos_handler.setup()
        self.pqos_handler.set_association_class()
        self.pqos_handler.print_association_config()

        # launch the load tester
        if self.mem_client is not None:
            self.__stop_client()
        self.__start_client()
        log.debug("Mem client started. Warm up period follows.")

        # collect tail latency before launching be
        for i in range(int(self.warm_up)):
            latency_list.append(get_latency())

        # launch the be container
        if self.container_be is not None:
            self.__stop_be()
        self.__start_be()
        log.debug('BE started')

        state = self.__get_next_state(self.action_space.n)

        return state

    def step(self, action_be_ways):
        """ At each step the agent specifies the number of ways that are assigned to the be"""

        done = False
        self.container_be.reload()
        if self.container_be.status == 'exited':
            self.container_be.remove()
            done = True

        err_msg = "%r (%s) invalid" % (action_be_ways, type(action_be_ways))
        assert self.action_space.contains(action_be_ways), err_msg

        # enforce the decision with PQOS
        self.pqos_handler.set_allocation_class(action_be_ways)
        self.pqos_handler.print_allocation_config()

        state = self.__get_next_state(action_be_ways)

        tail_latency = state[0]
        reward = self.__reward_func(action_be_ways, tail_latency)  # based on new metrics
        return state, reward, done, None

        # should we return done when the first app finishes ? or should we ignore this fact and just restart
        # episode could end after a number of steps
        # episode could end when we end up in a very bad state

    def render(self, **kwargs):
        pass

    def stop(self):
        log.debug('Stopping everything!')

        # check if container is running and stop it
        self.__stop_be()

        # wait a period of time after the collocation in order to collect metrics
        for i in range(int(self.warm_up)):
            latency_list.append(get_latency())

        # stop the mem client
        self.__stop_client()

        # stop pqos monitoring
        self.pqos_handler.stop()
        self.pqos_handler.reset()
        if self.pqos_interface != 'none':
            self.pqos.fini()

        latency_per = np.percentile(latency_list, 99)
        latency_list_per = [min(i, latency_per) for i in latency_list]
        plt.plot(latency_list_per)
        plt.title('Effect of collocation in tail latency')
        plt.axvline(x=self.warm_up, color='g', linestyle='dashed', label='BE starts')
        plt.axvline(x=len(latency_list_per) - self.warm_up, color='r', linestyle='dashed', label='BE stops')
        plt.axhline(y=self.latency_thr, color='m', label='Latency threshold')
        plt.xlabel('Steps')
        plt.ylabel('Q95 Latency in ms')
        plt.legend(loc='best')
        plt.savefig('runs/collocation_{}.png'.format(datetime.today().strftime('%Y%m%d_%H%M%S')))
        # plt.show()
