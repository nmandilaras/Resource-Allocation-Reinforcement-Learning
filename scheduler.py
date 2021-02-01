import time
import ast
import docker
import random
import logging.config
from utils.functions import parse_num_list
from utils.config_constants import *
from abc import ABC, abstractmethod


logging.config.fileConfig('logging.conf')  # NOTE path!
log = logging.getLogger('simpleExample')


class Scheduler:
    """ Handles all the operations needed to execute the Best Effort applications.
     Dockers containers are used to handle the execution. """

    def __init__(self, config_scheduler):
        self.cores_per_be = int(config_scheduler[CORES_PER_BE])
        self.cores_pids_be = config_scheduler[CORES_BE]
        self.cores_pids_be_range = parse_num_list(self.cores_pids_be)
        self.num_total_bes = int(config_scheduler[NUM_BES])
        self.container_bes = []
        self.start_time_bes = None
        self.stop_time_bes = None
        self.interval_bes = None  # in minutes
        self.client = docker.from_env()
        self.issued_bes = 0
        self.finished_bes = 0
        # self.cores_map = lambda i: ','.join(map(str, self.cores_pids_be_range[i * self.cores_per_be: (i + 1) * self.cores_per_be]))
        self.be_repeated = int(config_scheduler[BE_REPEATED])
        self.be_quota = self.be_repeated
        self.last_be = None
        self.new_be = False
        self.docker_file = config_scheduler[DOCKER_FILE]
        self.bes = self.read_avail_dockers()

    def cores_map(self, i):
        """  """
        cores_range = self.cores_pids_be_range[i * self.cores_per_be: (i + 1) * self.cores_per_be]
        cores_range_string = map(str, cores_range)
        return ','.join(cores_range_string)

    def read_avail_dockers(self):
        """ Gets a dictionary with the available BEs and their parameters needed for execution. """

        file = open(self.docker_file, "r")
        contents = file.read()
        bes = ast.literal_eval(contents)

        return bes

    @abstractmethod
    def _select_be(self):
        raise NotImplementedError

    def _repeat_be(self):
        """ Checks if a new be should be selected or the current one can be reintroduced """
        # ΝΟΤΕ do we still need this functionality for our experiments?
        # If yes, it can be implemented as decorator
        if self.be_quota >= self.be_repeated:
            self.new_be = True
            return self._select_be()
        else:
            self.be_quota += 1
            return self.last_be

    # def _select_be(self):
    #     """  """
    #     if self.be_quota == self.be_repeated:
    #         log.info("Quota expired new be will be issued!")
    #         self.be_quota = 1
    #         if self.be_name:
    #             next_be = self.be_name.pop(0)
    #             if next_be != self.last_be:
    #                 self.last_be = next_be
    #                 self.new_be = True
    #         else:
    #             self.generator.choice(list(self.bes.keys()))
    #             self.new_be = True
    #     else:
    #         self.be_quota += 1
    #
    #     return self.last_be

    def _start_be(self, cores):
        """ Start a container on specified cores. """

        # log.info('New BE will be issued on core(s): {} at step: {}'.format(cores, self.steps))

        be = self._select_be()
        log.info('Selected Job: {}'.format(be))
        container, command, volume = self.bes[be]
        container_be = self.client.containers.run(container, command=command, name='be_' + cores.replace(",", "_"),
                                                  cpuset_cpus=cores, volumes_from=[volume] if volume is not None else [],
                                                  detach=True)
        self.issued_bes += 1

        return container_be

    def start_bes(self):
        """ Launches bes. """

        num_startup_bes = len(self.cores_pids_be_range) // self.cores_per_be
        self.container_bes = [self._start_be(self.cores_map(i)) for i in range(num_startup_bes)]

        self.start_time_bes = time.time()

    def restart_bes(self, status):
        """ Issue new bes on cores that finished execution. """

        for i, status in enumerate(status):
            if status:
                self.container_bes[i] = self._start_be(self.cores_map(i))
                log.info("Finished Bes: {}/{}".format(self.finished_bes, self.num_total_bes))

    def poll_bes(self):
        """  """
        status = []
        for container_be in self.container_bes:
            container_be.reload()
            if container_be.status == 'exited':
                self._stop_be(container_be)
                status.append(True)
                self.finished_bes += 1
            else:
                status.append(False)

        # self.restart_bes(status)

        # done = self.finished_bes >= self.num_total_bes

        return status

    @staticmethod
    def _stop_be(container_be):
        """"""
        container_be.stop()
        container_be.remove()

    def stop_bes(self):
        for container_be in self.container_bes:
            self._stop_be(container_be)

    def determine_termination(self):
        status = self.poll_bes()
        self.restart_bes(status)
        done = self.finished_bes >= self.num_total_bes
        done = any(status)  # done if any of the bes has finished execution
        if done:
            self.stop_time_bes = time.time()
            self.interval_bes = (self.stop_time_bes - self.start_time_bes) / 60

        return done


class RandomScheduler(Scheduler):

    def __init__(self, config_scheduler):
        super().__init__(config_scheduler)
        self.generator = random.Random(int(config_scheduler[SEED]))

    def _select_be(self):
        return self.generator.choice(list(self.bes.keys()))


class ListScheduler(Scheduler):

    def __init__(self, config_scheduler):
        super().__init__(config_scheduler)
        self.be_name = ast.literal_eval(config_scheduler[BE_NAME])

    def _select_be(self):
        return self.be_name.pop(0)
