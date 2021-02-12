from enum import Enum

LC_TAG = "Latency Critical"
BE_TAG = "Best Effort"

metric_names = ['IPC', 'Misses per k. cycles', 'LLC Occupancy', 'Bandwidth L.', 'Bandwidth R.', 'Latency', 'RPS']


class Loaders(str, Enum):
    MEMCACHED = "memcached"


class Schedulers(str, Enum):
    RANDOM = "random"
    QUEUE = "queue"
