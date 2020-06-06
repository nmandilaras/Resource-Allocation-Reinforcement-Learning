from pqos import Pqos
from pqos.capability import PqosCap, CPqosMonitor
from pqos.cpuinfo import PqosCpuInfo
from pqos.monitoring import PqosMon
from pqos.l3ca import PqosCatL3
from pqos.allocation import PqosAlloc


ways = [0x00001, 0x00003, 0x00007, 0x0000f,
        0x0001f, 0x0003f, 0x0007f, 0x000ff,
        0x001ff, 0x003ff, 0x007ff, 0x00fff,
        0x01fff, 0x03fff, 0x07fff, 0x0ffff,
        0x1ffff, 0x3ffff, 0x7ffff, 0xfffff]

base = (1 << 20) - 1  # TODO consider auto-generate those by CpuInfo


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


class PqosContextManager:
    """
    Helper class for using PQoS library Python wrapper as a context manager
    (in with statement).
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pqos = Pqos()

    def __enter__(self):
        """Initializes PQoS library."""

        self.pqos.init(*self.args, **self.kwargs)
        return self.pqos

    def __exit__(self, *args, **kwargs):
        """Finalizes PQoS library."""

        self.pqos.fini()
        return None


class PqosHandler:
    """Generic class for monitoring"""

    def __init__(self, socket=0, cos_id_hp=0, cos_id_be=1):
        self.mon = PqosMon()
        self.alloc = PqosAlloc()
        self.l3ca = PqosCatL3()
        self.cap = PqosCap()
        self.cpu = PqosCpuInfo()
        self.socket = socket
        self.cos_id_hp = cos_id_hp
        self.cos_id_be = cos_id_be
        self.group_hp, self.group_be = None, None

    def get_supported_events(self):
        """
        Returns a list of supported monitoring events.

        Returns:
            a list of supproted monitor events
        """

        mon_cap = self.cap.get_type('mon')

        events = [get_event_name(event.type) for event in mon_cap.events]

        # Filter out perf events
        # events = list(filter(lambda event: 'perf' not in event, events))

        return events

    # def get_all_cores(self):
    #     """
    #     Returns all available pids_bes.
    #
    #     Returns:
    #         a list of available pids_bes
    #     """
    #
    #     cores = []
    #     sockets = self.cpu.get_sockets()
    #
    #     for socket in sockets:
    #         cores += self.cpu.get_cores(socket)
    #
    #     return cores

    def setup_groups(self):
        """Sets up monitoring groups. Needs to be implemented by a derived class."""

        return []

    def setup(self):
        """Resets monitoring and configures (starts) monitoring groups."""

        self.mon.reset()
        self.group_hp, self.group_be = self.setup_groups()

    def update(self):
        """Updates values for monitored events."""

        self.mon.poll([self.group_hp, self.group_be])

    def print_data(self):
        """Prints current values for monitored events. Needs to be implemented
        by a derived class."""

        pass

    def stop(self):
        """Stops monitoring."""

        self.group_hp.stop()
        self.group_be.stop()

    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs or PIDs
        """
        pass

    def set_allocation_class(self, ways_be):
        """
        Sets up allocation classes of service on selected CPU sockets

        Parameters:
            ways_be: num of ways to be assigned for bes
        """
        mask_be = ways[ways_be - 1]
        mask_hp = mask_be ^ base
        cos_hp = self.l3ca.COS(self.cos_id_hp, mask_hp)
        cos_be = self.l3ca.COS(self.cos_id_be, mask_be)

        try:
            self.l3ca.set(self.socket, [cos_hp, cos_be])
        except:
            print("Setting up cache allocation class of service failed!")

    def reset_allocation(self):
        """ Resets allocation configuration. """

        try:
            self.alloc.reset('any', 'any', 'any')
            print("Allocation reset successful")
        except:
            print("Allocation reset failed!")


class PqosHandlerCore(PqosHandler):
    """PqosHandler per core"""

    def __init__(self, cores_hp, cores_be, events):
        """
        Initializes object of this class with cores and events to monitor.

        Parameters:
            cores_hp: a list of cores assigned to hp
            cores_be: a list of cores assigned to bes
            events: a list of monitoring events
        """

        super(PqosHandlerCore, self).__init__()
        self.cores_hp = cores_hp
        self.cores_be = cores_be
        self.events = events

    def setup_groups(self):
        """
        Starts monitoring for each core using separate monitoring groups for each core.

        Returns:
            created monitoring groups
        """

        group_hp = self.mon.start(self.cores_hp, self.events)
        group_be = self.mon.start(self.cores_be, self.events)

        return group_hp, group_be

    def print_data(self):
        """Prints current values for monitored events."""

        print("    CORE    RMID     IPC    MISSES    LLC[KB]    MBL[MB]    MBR[MB]")

        for group in self.groups:
            core = group.cores[0]
            # rmid = group.poll_ctx[0].rmid if group.poll_ctx else 'N/A'
            rmid = -1  # changed by nikmand
            ipc = group.values.ipc
            misses = group.values.llc_misses
            llc = bytes_to_kb(group.values.llc)
            mbl = bytes_to_mb(group.values.mbm_local_delta)
            mbr = bytes_to_mb(group.values.mbm_remote_delta)
            print("%8u %8s %6.2f %8.1f %10.1f %10.1f %10.1f" % (core, rmid, ipc, misses, llc, mbl, mbr))

    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            cores: a list of cores
        """

        try:
            for core_hp in self.group_hp.cores:
                self.alloc.assoc_set(core_hp, self.cos_id_hp)
            for core_be in self.group_be.cores:
                self.alloc.assoc_set(core_be, self.cos_id_be)
        except:
            print("Setting allocation class of service association failed!")


class PqosHandlerPid(PqosHandler):
    """PqosHandler per PID (OS interface only)"""

    def __init__(self, pid_hp, pids_be, events):
        """
        Initializes object of this class with PIDs and events to monitor.

        Parameters:
            pid_hp: pid of hp
            pids_be: a list of PIDs to monitor
            events: a list of monitoring events
        """

        super(PqosHandlerPid, self).__init__()
        self.pid_hp = pid_hp
        self.pids_be = pids_be
        self.events = events

    def setup_groups(self):
        """
        Starts monitoring for each PID using separate monitoring groups for
        each PID.

        Returns:
            created monitoring groups
        """

        # NOTE there is the ability to add/remove pids_be to/from a group

        group_hp = self.mon.start_pids([self.pid_hp], self.events)
        group_be = self.mon.start_pids(self.pids_be, self.events)

        return group_hp, group_be

    def print_data(self):
        """Prints current values for monitored events."""

        print("   PID    LLC[KB]    MBL[MB]    MBR[MB]")

        pid = 'lc_critical'
        values_hp = self.group_hp.values
        ipc = values_hp.ipc
        misses = values_hp.llc_misses
        llc = bytes_to_kb(values_hp.llc)
        mbl = bytes_to_mb(values_hp.mbm_local_delta)
        mbr = bytes_to_mb(values_hp.values.mbm_remote_delta)
        print("%6d %10.1f %10.1f %10.1f" % (pid, llc, mbl, mbr))

        # TODO bes

    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            pids: a list of pids_bes
        """

        try:
            self.alloc.assoc_set_pid(self.group_hp.group.pids[0], self.cos_id_hp)
            for pid in self.group_be.pids:
                self.alloc.assoc_set_pid(pid, self.cos_id_be)
        except:
            print("Setting allocation class of service association failed!")
