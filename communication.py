import socket
import struct
import ctypes
import subprocess
from time import sleep
import logging.config
from abc import ABC, abstractmethod
from utils.config_constants import *

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 42171      # The port used by the server

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')


class Loader(ABC):
    """ Abstract class that handles all the functionality that concerns the service loader. """
    def __init__(self, config_loader):
        self.mem_client = None
        self.service_ip = config_loader[HP_IP]
        self.service_port = config_loader[HP_PORT]
        self.loader_dir = config_loader[LOADER_DIR]
        self.quantile = config_loader[QUANTILE]
        self.measurement_interval = config_loader[ACTION_INTERVAL]
        self.rps = int(config_loader[LOADER_RPS])

    @abstractmethod
    def start(self):
        """ Starts loader as a subprocess. """
        raise NotImplementedError

    def stop(self):
        """ Sends signal to stop the loader and checks for proper termination """

        self.mem_client.terminate()
        sleep(0.5)
        while self.mem_client.poll() is None:
            log.debug("Unable to shutdown mem client. Retrying...")
            self.mem_client.terminate()

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
    def __init__(self, config_loader):
        super().__init__(config_loader)
        self.cores_loader = config_loader[CORES_LOADER]
        self.loader_threads = config_loader[LOADER_THREADS]
        self.loader_conn = config_loader[LOADER_CONN]
        self.ratio = config_loader[GET_SET_RATIO]
        self.exponential_dist = config_loader[EXP_DIST]

    def start(self):
        """ Starts memcached loader with all necessary args. """

        loader = '{}/loader'.format(self.loader_dir)
        dataset = '{}/twitter_dataset/twitter_dataset_30x'.format(self.loader_dir)
        servers = '{}/docker_servers.txt'.format(self.loader_dir)
        self.mem_client = subprocess.Popen(['taskset', '--cpu-list', self.cores_loader, loader, '-a', dataset,
                                            '-s', servers, '-g', self.ratio, '-c', self.loader_conn, '-w',
                                            self.loader_threads, '-T', self.measurement_interval, '-r', str(self.rps),
                                            '-q', self.quantile, self.exponential_dist])
        sleep(10)  # wait in order to bind the socket


def get_loader_stats():
    """  """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b'get q95')

        fmt = "dd"
        fmt_size = struct.calcsize(fmt)
        data = s.recv(fmt_size)        # this call will block
        latency, rps = struct.unpack(fmt, data[:fmt_size])

    log.debug('Tail latency q95: {}'.format(latency))
    log.debug('RPS: {}'.format(rps))

    return latency, rps
