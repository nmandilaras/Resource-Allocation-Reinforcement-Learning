import socket
import struct
import ctypes
import logging.config

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 42171      # The port used by the server

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')


# class Stats(ctypes.Structure):
#     _fields_ = [
#         ("q95", ctypes.c_double),
#         ("rps", ctypes.c_double)
#     ]


def get_loader_stats():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # use unix sockets?
        s.connect((HOST, PORT))
        s.sendall(b'get q95')

        # stats = Stats()
        fmt = "d"
        fmt_size = struct.calcsize(fmt)
        # log.debug('fmt_size:{}'.format(fmt_size))
        data = s.recv(fmt_size)        # this call will block
        q95 = struct.unpack(fmt, data[:fmt_size])[0]

    log.debug('Tail latency q95: {}'.format(q95))
    # log.debug('RPS: {}'.format(rps))

    return q95, -1
