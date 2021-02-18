import socket
import struct
import subprocess
from time import sleep
import logging.config
from abc import ABC, abstractmethod
from utils.config_constants import *

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')


class Loader(ABC):
    """ Abstract class that handles all the functionality that concerns the service loader. """
    def __init__(self, config):
        self.client = None
        self.service_ip = config[HP_IP]
        self.service_port = config.getint(HP_PORT)
        self.loader_dir = config[LOADER_DIR]
        self.quantile = config[QUANTILE]
        self.measurement_interval = config[ACTION_INTERVAL]
        self.rps = config.getint(LOADER_RPS)
        self.cores_loader = config[CORES_LOADER]

    @abstractmethod
    def start(self):
        """ Starts loader as a subprocess. """
        raise NotImplementedError

    def stop(self):
        """ Sends signal to stop the loader and checks for proper termination """

        self.client.terminate()
        sleep(0.5)
        while self.client.poll() is None:
            log.debug("Unable to shutdown loader. Retrying...")
            self.client.terminate()

    def reset(self):
        """ Restart the loader. """

        if self.client is not None:
            self.stop()
        self.start()

    def get_stats(self):
        """ Collects the stats from the loader. Currently we are receiving the specified quantile
        and the requests per second. """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.service_ip, self.service_port))
            s.sendall(b'get q95')  # The text can be anything it just unblocks the loader

            fmt = "dd"
            fmt_size = struct.calcsize(fmt)
            data = s.recv(fmt_size)  # this call will block
            latency, rps = struct.unpack(fmt, data[:fmt_size])

        # log.debug('Tail latency {}: {}'.format(self.quantile, latency))
        # log.debug('RPS: {}'.format(rps))

        return latency, rps


class MemCachedLoader(Loader):
    """ Wrapper class for Memcached loader. """
    def __init__(self, config):
        super().__init__(config)
        self.loader_threads = config[LOADER_THREADS]
        self.loader_conn = config[LOADER_CONN]
        self.ratio = config[GET_SET_RATIO]
        self.exponential_dist = config[EXP_DIST]

    def start(self):
        """ Starts memcached loader with all necessary args. """
        # TODO probably subprocess could be moved to superclass and only arguments should be defined in subclasses
        loader = '{}/loader'.format(self.loader_dir)
        dataset = '{}/twitter_dataset/twitter_dataset_30x'.format(self.loader_dir)
        servers = '{}/docker_servers.txt'.format(self.loader_dir)
        self.client = subprocess.Popen(['taskset', '--cpu-list', self.cores_loader, loader, '-a', dataset, '-s',
                                        servers, '-g', self.ratio, '-c', self.loader_conn, '-w', self.loader_threads,
                                        '-T', self.measurement_interval, '-r', str(self.rps),  '-q', self.quantile,
                                        self.exponential_dist])
        sleep(10)  # wait in order to bind the socket

        log.debug("Loader started.")
