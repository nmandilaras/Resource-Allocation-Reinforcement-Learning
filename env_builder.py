from loader import MemCachedLoader
from pqos_handler import PqosHandlerMock, PqosHandlerPid, PqosHandlerCore
from rdt_env import Rdt
from scheduler import RandomScheduler, QueueScheduler
from utils.constants import Loaders, Schedulers


def loader_factory(service_name, config):
    """  """
    if service_name == Loaders.MEMCACHED:
        loader = MemCachedLoader(config)
    else:
        raise ValueError("Loader option {} is not supported".format(service_name))

    return loader


def scheduler_factory(scheduler_type, config):
    """  """
    if scheduler_type == Schedulers.RANDOM:
        scheduler = RandomScheduler(config)
    elif scheduler_type == Schedulers.QUEUE:
        scheduler = QueueScheduler(config)
    else:
        raise ValueError("Scheduler option {} is not supported".format(scheduler_type))

    return scheduler


def pqos_factory(pqos_interface, cores_pid_hp_range, cores_pids_be_range):
    """  """
    if pqos_interface == 'MSR':
        pqos_handler = PqosHandlerCore(cores_pid_hp_range, cores_pids_be_range)
    elif pqos_interface == 'OS':
        pqos_handler = PqosHandlerPid(cores_pid_hp_range, cores_pids_be_range)
    else:
        pqos_handler = PqosHandlerMock()

    return pqos_handler


class EnvBuilder:
    """ It takes over the creation of the environment. """

    def __init__(self):
        self.loader = None
        self.scheduler = None
        self.pqos_handler = None

    def build_loader(self, service_name, config):
        self.loader = loader_factory(service_name, config)

        return self

    def build_pqos(self, pqos_interface):
        # self.pqos_handler = pqos_factory(pqos_interface, ,)

        return self

    def build_scheduler(self, scheduler_type, config):
        self.scheduler = scheduler_factory(scheduler_type, config)

        return self

    def build(self, config):
        env = Rdt(config, self.pqos_handler, self.scheduler, self.scheduler)

        return env
