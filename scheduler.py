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


def read_avail_dockers(docker_file):
    """ Gets a dictionary with the available BEs and their parameters needed for execution. """

    file = open(docker_file, "r")
    contents = file.read()
    bes = ast.literal_eval(contents)

    return bes


class Scheduler:
    """ Handles all the operations needed to execute the Best Effort applications.
     Dockers containers are used to handle the execution. """

    def __init__(self, config_scheduler):
        self.cores_per_be = int(config_scheduler[CORES_PER_BE])
        self.cores_pids_be_range = parse_num_list(config_scheduler[CORES_BE])
        self.container_bes = []
        self.client = docker.from_env()

        self.finished_bes = 0
        self.bes_available = read_avail_dockers(config_scheduler[DOCKER_FILE])

        self.issued_bes = 0  # what is it used for ?
        self.be_repeated = int(config_scheduler[BE_REPEATED])
        self.be_quota = self.be_repeated  # there are set equal so that in the first check a new BE will be issued
        self.last_be = None
        self.new_be = False
        self.num_total_bes = int(config_scheduler[NUM_BES])

        self.start_time_bes = None
        self.stop_time_bes = None
        self.experiment_duration = None  # in minutes

    def cores_map(self, i):
        """ Returns the cores that corresponds to the ith container. """
        cores_range = self.cores_pids_be_range[i * self.cores_per_be: (i + 1) * self.cores_per_be]
        cores_range_string = map(str, cores_range)
        return ','.join(cores_range_string)

    @abstractmethod
    def _select_be(self):
        raise NotImplementedError

    def _repeat_be(self):
        """ Checks if a new be should be selected or the current one can be reintroduced """
        # ΝΟΤΕ do we still need this functionality for our experiments?
        # If yes, it can be implemented as decorator
        if self.be_quota >= self.be_repeated:
            self.be_quota = 1
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
        container, command, volume = self.bes_available[be]
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

    def restart_bes(self, have_finished):
        """ Issue new bes on cores that finished execution, if there are any. """

        for i, has_finished in enumerate(have_finished):
            if has_finished:
                self._stop_be(self.container_bes[i])
                self.container_bes[i] = self._start_be(self.cores_map(i))
                log.info("Finished Bes: {}/{}".format(self.finished_bes, self.num_total_bes))

    def poll_bes(self):
        """ Reloads the status of containers and checks if they have exited. """
        have_finished = []
        for container_be in self.container_bes:
            container_be.reload()
            if container_be.status == 'exited':
                have_finished.append(True)
                self.finished_bes += 1
            else:
                have_finished.append(False)

        return have_finished

    @staticmethod
    def _stop_be(container_be):
        """ Stops and removes exited containers.  """
        container_be.stop()
        container_be.remove()

    def stop_bes(self):
        """ Stops all the containers. """
        for container_be in self.container_bes:
            self._stop_be(container_be)

    def update_status(self):
        """ Polls the status of the containers and determines which of them have finished. If the  """
        have_finished = self.poll_bes()
        done = self.finished_bes >= self.num_total_bes
        # done = any(have_finished)  # done if any of the bes has finished execution
        if done:
            self.stop_time_bes = time.time()
            self.experiment_duration = (self.stop_time_bes - self.start_time_bes) / 60
        else:
            self.restart_bes(have_finished)

        return done

    def get_experiment_duration(self):
        """ Returns the time needed to for the bes to be completed. """

        minutes = self.experiment_duration
        seconds = int(round((self.experiment_duration % 1) * 60, 0))
        duration = str(minutes) + 'm' + str(seconds) + 's'

        return duration


class RandomScheduler(Scheduler):
    """ Initializes a random generator given a specific seed. The choices of the bes are made by the generator. """

    def __init__(self, config_scheduler):
        super().__init__(config_scheduler)
        self.generator = random.Random(int(config_scheduler[SEED]))

    def _select_be(self):
        return self.generator.choice(list(self.bes_available.keys()))


class ListScheduler(Scheduler):
    """  """

    def __init__(self, config_scheduler):
        super().__init__(config_scheduler)
        self.bes_selected = ast.literal_eval(config_scheduler[BES_LIST])

    def _select_be(self):
        return self.bes_selected.pop(0)
