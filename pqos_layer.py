from pqos import Pqos
from pqos.capability import PqosCap, CPqosMonitor
from pqos.cpuinfo import PqosCpuInfo
from pqos.monitoring import PqosMon
from pqos.l3ca import PqosCatL3
from pqos.allocation import PqosAlloc


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
        "Initializes PQoS library."

        self.pqos.init(*self.args, **self.kwargs)
        return self.pqos

    def __exit__(self, *args, **kwargs):
        "Finalizes PQoS library."

        self.pqos.fini()
        return None


class Monitoring:
    """Generic class for monitoring"""

    def __init__(self, socket=0, cos_id_hp=0, cos_id_be=1):
        self.mon = PqosMon()
        self.alloc = PqosAlloc()
        self.l3ca = PqosCatL3()
        self.socket = socket
        self.cos_id_hp = cos_id_hp
        self.cos_id_be = cos_id_be
        self.groups = []

    def setup_groups(self):
        """Sets up monitoring groups. Needs to be implemented by a derived class."""

        return []

    def setup(self):
        """Resets monitoring and configures (starts) monitoring groups."""

        self.mon.reset()
        self.groups = self.setup_groups()

    def update(self):
        """Updates values for monitored events."""

        self.mon.poll(self.groups)

    def print_data(self):
        """Prints current values for monitored events. Needs to be implemented
        by a derived class."""

        pass

    def stop(self):
        """Stops monitoring."""

        for group in self.groups:
            group.stop()

    def set_association_class(self, class_id, cores):
        """
        Sets up allocation classes of service on selected CPUs or PIDs

        Parameters:
            class_id: class of service ID
            cores: a list of pids
        """
        pass

    def set_allocation_class(self, mask_hp, mask_be):
        """
        Sets up allocation classes of service on selected CPU sockets

        Parameters:
            mask_hp: COS bitmask for hp
            mask_be: COS bitmask for bes
        """

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


class MonitoringCore(Monitoring):
    """Monitoring per core"""

    def __init__(self, cores, events):
        """
        Initializes object of this class with pids and events to monitor.

        Parameters:
            cores: a list of pids to monitor
            events: a list of monitoring events
        """

        super(MonitoringCore, self).__init__()
        self.cores = cores or get_all_cores()
        self.events = events

    def setup_groups(self):
        """
        Starts monitoring for each core using separate monitoring groups for
        each core.

        Returns:
            created monitoring groups
        """

        groups = []

        for core in self.cores:
            group = self.mon.start([core], self.events)
            groups.append(group)

        return groups

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

    def set_association_class(self, class_id, cores):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            cores: a list of cores
        """

        for core in cores:
            try:
                self.alloc.assoc_set(core, class_id)
            except:
                print("Setting allocation class of service association failed!")


class MonitoringPid(Monitoring):
    """Monitoring per PID (OS interface only)"""

    def __init__(self, pids, events):
        """
        Initializes object of this class with PIDs and events to monitor.

        Parameters:
            pids: a list of PIDs to monitor
            events: a list of monitoring events
        """

        super(MonitoringPid, self).__init__()
        self.pids = pids
        self.events = events

    def setup_groups(self):
        """
        Starts monitoring for each PID using separate monitoring groups for
        each PID.

        Returns:
            created monitoring groups
        """

        groups = []

        for pid in self.pids:
            group = self.mon.start_pids([pid], self.events)
            groups.append(group)

        return groups

    def print_data(self):
        "Prints current values for monitored events."

        print("   PID    LLC[KB]    MBL[MB]    MBR[MB]")

        for group in self.groups:
            pid = group.pids[0]
            llc = bytes_to_kb(group.values.llc)
            mbl = bytes_to_mb(group.values.mbm_local_delta)
            mbr = bytes_to_mb(group.values.mbm_remote_delta)
            print("%6d %10.1f %10.1f %10.1f" % (pid, llc, mbl, mbr))

    def set_association_class(self, class_id, pids):
        """
        Sets up allocation classes of service on selected CPUs

        Parameters:
            class_id: class of service ID
            pids: a list of pids
        """

        for pid in pids:
            try:
                self.alloc.assoc_set_pid(pid, class_id)
            except:
                print("Setting allocation class of service association failed!")
