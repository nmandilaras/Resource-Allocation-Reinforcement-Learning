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


def get_metrics(group_values):

    ipc = group_values.ipc
    misses = group_values.llc_misses
    llc = bytes_to_kb(group_values.llc)
    mbl = bytes_to_mb(group_values.mbm_local_delta)
    mbr = bytes_to_mb(group_values.mbm_remote_delta)

    return ipc, misses, llc, mbl, mbr


class PqosHandler:
    """Generic class for monitoring"""

    def __init__(self, socket=0, cos_id_hp=0, cos_id_be=1):
        # self.pqos = Pqos()
        # self.pqos.init(interface)
        self.mon = PqosMon()
        self.alloc = PqosAlloc()
        self.l3ca = PqosCatL3()
        self.cap = PqosCap()
        self.cpu_info = PqosCpuInfo()
        self.socket = socket
        self.cos_id_hp = cos_id_hp
        self.cos_id_be = cos_id_be
        self.group_hp, self.group_be = None, None

    def finish(self):
        pass
        # self.pqos.fini()
        # self.stop()

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

    def get_all_cores(self):
        """
        Returns a list of all available cores
        """

        cores = []
        sockets = self.cpu_info.get_sockets()

        for socket in sockets:
            cores += self.cpu_info.get_cores(socket)

        return cores

    def setup_groups(self):
        """Sets up monitoring groups. Needs to be implemented by a derived class."""

        return []

    def reset(self):
        """Resets monitoring and configures (starts) monitoring groups."""

        self.mon.reset()
        self.__reset_allocation_association()

    def update(self):
        """Updates values for monitored events."""

        self.mon.poll([self.group_hp, self.group_be])

    def get_hw_metrics(self):
        """Prints current values for monitored events."""

        print("    CORE     IPC    MISSES    LLC[KB]    MBL[MB]    MBR[MB]")

        ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp = get_metrics(self.group_hp.values)
        ipc_be, misses_be, llc_be, mbl_be, mbr_be = get_metrics(self.group_be.values)

        print("%8s %6.2f %8.1f %10.1f %10.1f %10.1f" % ('lc_critical', ipc_hp, misses_hp, llc_hp, mbl_hp, mbr_hp))
        print("%8s %6.2f %8.1f %10.1f %10.1f %10.1f" % ('bes', ipc_be, misses_be, llc_be, mbl_be, mbr_be))

        socket_wide_bw = mbl_hp + mbl_be

        return misses_be, socket_wide_bw

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

    def print_association_config(self):
        pass

    def print_allocation_config(self):
        sockets = self.cpu_info.get_sockets()
        for socket in sockets:
            try:
                coses = self.l3ca.get(socket)

                print("L3CA COS definitions for Socket %u:" % socket)

                for cos in coses:
                    cos_params = (cos.class_id, cos.mask)
                    print("    L3CA COS%u => MASK 0x%x" % cos_params)
            except:
                print("Error in getting allocation configuration")
                raise

    def __reset_allocation_association(self):
        """ Resets allocation and association configuration. """

        try:
            self.alloc.reset('any', 'any', 'any')
            print("Allocation reset successful")
        except:
            print("Allocation reset failed!")


class PqosHandlerCore(PqosHandler):
    """PqosHandler per core"""

    def __init__(self, cores_hp, cores_be):
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
        self.events = self.get_supported_events()
        self.group_hp, self.group_be = self.setup_groups()

    def setup_groups(self):
        """
        Starts monitoring for each core using separate monitoring groups for each core.

        Returns:
            created monitoring groups
        """

        group_hp = self.mon.start(self.cores_hp, self.events)
        group_be = self.mon.start(self.cores_be, self.events)

        return group_hp, group_be

    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            cores: a list of cores
        """

        try:
            for core_hp in self.cores_hp:
                self.alloc.assoc_set(core_hp, self.cos_id_hp)
            for core_be in self.cores_be:
                self.alloc.assoc_set(core_be, self.cos_id_be)
        except:
            print("Setting allocation class of service association failed!")

    def print_association_config(self):
        """"""
        cores = self.get_all_cores()
        for core in cores:
            class_id = self.alloc.assoc_get(core)
            print("Core %u => COS%u" % (core, class_id))


class PqosHandlerPid(PqosHandler):
    """PqosHandler per PID (OS interface only)"""

    def __init__(self, pid_hp, pids_be):
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
        self.events = self.get_supported_events()
        self.group_hp, self.group_be = self.setup_groups()

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

    def set_association_class(self):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            pids: a list of pids_bes
        """

        try:
            # TODO probably this have to change pid list instead of group_hp
            self.alloc.assoc_set_pid(self.group_hp.group.pids[0], self.cos_id_hp)
            for pid in self.group_be.pids:
                self.alloc.assoc_set_pid(pid, self.cos_id_be)
        except:
            print("Setting allocation class of service association failed!")

    def print_association_config(self):
        pids = [self.pid_hp] + self.pids_be
        for pid in pids:
            class_id = self.alloc.assoc_get_pid(pid)
            print("Pid %u => COS%u" % (pid, class_id))