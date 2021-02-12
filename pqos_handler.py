from pqos import Pqos
from pqos.capability import PqosCap, CPqosMonitor
from pqos.cpuinfo import PqosCpuInfo
from pqos.monitoring import PqosMon
from pqos.l3ca import PqosCatL3
from pqos.allocation import PqosAlloc
import logging.config
from random import randint, randrange
from abc import ABC, abstractmethod

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

# NOTE we define all possible masks for our server which has 20 ways in LLC (L3)
# Due to pqos limitation on the left side the min value of ways is 2 so HP service will always have at least two ways

L3_NUM_WAYS = 20  # NOTE consider getting this number by CpuInfo

# ways that can be assigned to BEs
ways = [0x00001, 0x00003, 0x00007, 0x0000f,
        0x0001f, 0x0003f, 0x0007f, 0x000ff,
        0x001ff, 0x003ff, 0x007ff, 0x00fff,
        0x01fff, 0x03fff, 0x07fff, 0x0ffff,
        0x1ffff, 0x3ffff, 0x7ffff, 0xfffff]

# ways = [(1 << i) - 1 for i in range(1, L3_NUM_WAYS + 1)]

base = (1 << L3_NUM_WAYS) - 1  #

# base = ways[-1]


def bytes_to_kb(num_bytes):
    """
    Converts bytes to kilobytes.

    :param num_bytes: number of bytes
    :return: number of kilobytes
    """

    return num_bytes / 1024.0


def bytes_to_mb(num_bytes):
    """
    Converts bytes to megabytes.

    :param num_bytes: number of bytes
    :returns: number of megabytes
    """

    return num_bytes / (1024.0 * 1024.0)


def get_event_name(event_type):
    """
    Converts a monitoring event type to a string label required by libpqos Python wrapper.

    :param event_type: monitoring event type
    :return: a string label
    """

    event_map = {
        CPqosMonitor.PQOS_MON_EVENT_L3_OCCUP: 'l3_occup',
        CPqosMonitor.PQOS_MON_EVENT_LMEM_BW: 'lmem_bw',
        CPqosMonitor.PQOS_MON_EVENT_TMEM_BW: 'tmem_bw',
        CPqosMonitor.PQOS_MON_EVENT_RMEM_BW: 'rmem_bw',
        CPqosMonitor.PQOS_PERF_EVENT_LLC_MISS: 'perf_llc_miss',
        CPqosMonitor.PQOS_PERF_EVENT_IPC: 'perf_ipc'
    }

    return event_map.get(event_type)


def get_metrics(group_values, time_interval):
    """  """
    ipc = group_values.ipc
    misses = group_values.llc_misses_delta  # / (group_values.ipc_unhalted_delta / 1000.)

    llc = bytes_to_mb(group_values.llc)
    mbl = bytes_to_mb(group_values.mbm_local_delta)
    mbr = bytes_to_mb(group_values.mbm_remote_delta)

    cycles = group_values.ipc_unhalted_delta
    instructions = group_values.ipc_retired_delta

    mbl_ps, mbr_ps = mbl / time_interval, mbr / time_interval

    return ipc, misses, llc, mbl_ps, mbr_ps, cycles, instructions


def get_metrics_random():
    """ Mock method that returns same arguments as the func get_metrics """
    ipc = randrange(0, 2)
    misses = randint(1e3, 1e5)
    llc = randint(1e3, 1e5)
    mbl = randint(1e2, 1e3)
    mbr = randint(1e2, 1e3)
    cycles = randint(1e2, 1e3)
    instructions = randint(1e2, 1e3)

    return ipc, misses, llc, mbl, mbr, cycles, instructions


class PqosHandler(ABC):
    """ Generic class for monitoring """

    def __init__(self, interface, socket=0, cos_id_hp=1, cos_id_be=2):
        self.pqos = Pqos()
        self.pqos.init(interface)
        self.mon = PqosMon()
        self.alloc = PqosAlloc()
        self.l3ca = PqosCatL3()
        self.cap = PqosCap()
        self.cpu_info = PqosCpuInfo()
        self.socket = socket  # The experiment takes place at a signle socket
        self.cos_id_hp = cos_id_hp
        self.cos_id_be = cos_id_be
        self.group_hp, self.group_be = None, None
        self.events = self.get_supported_events()

    @abstractmethod
    def setup_groups(self):  # NOTE this MUST follow reset of monitoring
        """Sets up monitoring groups. Needs to be implemented by a derived class."""
        raise NotImplementedError

    @abstractmethod
    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs or PIDs
        """
        raise NotImplementedError

    @abstractmethod
    def print_association_config(self):
        """  """
        raise NotImplementedError

    def finish(self):
        self.pqos.fini()

    def get_supported_events(self):
        """ Returns a list of supported monitoring events. """

        mon_cap = self.cap.get_type('mon')

        events = [get_event_name(event.type) for event in mon_cap.events]

        # Filter out perf events
        # events = list(filter(lambda event: 'perf' not in event, events))

        return events

    def get_all_cores(self):
        """ Returns a list of all available cores. Used for informational reasons only. """

        cores = []
        sockets = self.cpu_info.get_sockets()

        for socket in sockets:
            cores += self.cpu_info.get_cores(socket)

        return cores

    def reset(self):
        """ Resets monitoring and configures (starts) monitoring groups. """

        self.mon.reset()
        self.reset_allocation_association()

    def update(self):
        """ Updates values for monitored events. """

        self.mon.poll([self.group_hp, self.group_be])

    def get_hp_metrics(self, time_interval):
        return get_metrics(self.group_hp.values, time_interval)

    def get_be_metrics(self, time_interval):
        return get_metrics(self.group_be.values, time_interval)

    def stop(self):
        """ Stops monitoring."""

        self.group_hp.stop()
        self.group_be.stop()

    def set_allocation_class(self, ways_be):
        """
        Sets up allocation classes of service on selected CPU sockets

        Parameters:
            ways_be: num of ways to be assigned for bes
        """
        if ways_be == -1:  # default setting, all ways can be accessed by both groups
            mask_be = ways[-1]
            mask_hp = ways[-1]
        else:
            mask_be = ways[ways_be]
            mask_hp = mask_be ^ base
        cos_hp = self.l3ca.COS(self.cos_id_hp, mask_hp)
        cos_be = self.l3ca.COS(self.cos_id_be, mask_be)

        try:
            self.l3ca.set(self.socket, [cos_hp, cos_be])
        except:
            log.error("Setting up cache allocation class of service failed!")
            raise

    def print_allocation_config(self):
        """  """
        sockets = [self.socket]  # self.cpu_info.get_sockets()
        for socket in sockets:
            try:
                coses = self.l3ca.get(socket)

                log.debug("L3CA COS definitions for Socket %u:" % socket)

                for cos in coses:
                    if cos.class_id == self.cos_id_be or cos.class_id == self.cos_id_hp:
                        cos_params = (cos.class_id, cos.mask)
                        log.debug("    L3CA COS%u => MASK 0x%x" % cos_params)
            except:
                log.warning("Error in getting allocation configuration")
                raise

    def reset_allocation_association(self):
        """ Resets allocation and association configuration. """

        try:
            self.alloc.reset('any', 'any', 'any')
            log.debug("Allocation reset successful")
        except:
            log.warning("Allocation reset failed!")
            raise


class PqosHandlerCore(PqosHandler):
    """ PqosHandler per core. """

    def __init__(self, cores_hp, cores_be):
        """
        Initializes object of this class with cores and events to monitor.

        Parameters:
            cores_hp: a list of cores assigned to hp
            cores_be: a list of cores assigned to bes
        """

        interface = "MSR"
        super(PqosHandlerCore, self).__init__(interface)
        self.cores_hp = cores_hp
        self.cores_be = cores_be

    def setup_groups(self):
        """ Starts monitoring for each group of cores. """

        self.group_hp = self.mon.start(self.cores_hp, self.events)
        self.group_be = self.mon.start(self.cores_be, self.events)

    def set_association_class(self):
        """ Sets up allocation classes of service on selected CPUs. """

        try:
            for core_hp in self.cores_hp:
                self.alloc.assoc_set(core_hp, self.cos_id_hp)
            for core_be in self.cores_be:
                self.alloc.assoc_set(core_be, self.cos_id_be)
        except:
            log.error("Setting association between core and class of service failed!")
            raise

    def print_association_config(self):
        """  """
        cores = self.cores_hp + self.cores_be  # or self.get_all_cores()
        for core in cores:
            class_id = self.alloc.assoc_get(core)
            log.debug("Core %u => COS%u" % (core, class_id))


class PqosHandlerPid(PqosHandler):
    """ PqosHandler per PID (OS interface only). """

    def __init__(self, pid_hp, pids_be):
        """
        Initializes object of this class with PIDs and events to monitor.

        Parameters:
            pid_hp: pid of hp
            pids_be: a list of PIDs to monitor
        """

        interface = "OS"
        super(PqosHandlerPid, self).__init__(interface)
        self.pid_hp = pid_hp
        self.pids_be = pids_be

    def setup_groups(self):
        """ Starts monitoring for group of PID(s). """

        # NOTE there is the ability to add/remove pids_be to/from a group

        self.group_hp = self.mon.start_pids([self.pid_hp], self.events)
        self.group_be = self.mon.start_pids(self.pids_be, self.events)

    def set_association_class(self):
        """ Sets up association classes of service on hp pid as well as in be pids. """

        try:
            self.alloc.assoc_set_pid(self.pid_hp, self.cos_id_hp)
            for pid in self.pids_be:
                self.alloc.assoc_set_pid(pid, self.cos_id_be)
        except:
            log.error("Setting association between pid and class of service failed!")
            raise

    def print_association_config(self):
        """  """
        pids = [self.pid_hp] + self.pids_be
        for pid in pids:
            class_id = self.alloc.assoc_get_pid(pid)
            log.debug("Pid %u => COS%u" % (pid, class_id))


class PqosHandlerMock:
    """ Mock class for use in environments where pqos cannot be installed. """

    def __int__(self, socket=0, cos_id_hp=1, cos_id_be=2):
        pass

    def setup_groups(self):
        pass

    def reset(self):
        pass

    def update(self):
        pass

    def get_hp_metrics(self, time_interval):
        return get_metrics_random()

    def get_be_metrics(self, time_interval):
        return get_metrics_random()

    def stop(self):
        pass

    def set_association_class(self):
        pass

    def set_allocation_class(self, ways_be):
        pass

    def reset_allocation_association(self):
        pass

    def print_association_config(self):
        pass

    def print_allocation_config(self):
        pass

    def finish(self):
        pass
