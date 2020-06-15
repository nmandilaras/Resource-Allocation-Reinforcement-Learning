import socket
import struct
import logging.config

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8080        # The port used by the server

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')


def get_latency():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # use unix sockets?
        s.connect((HOST, PORT))
        s.sendall(b'get q95')
        data = s.recv(8)
        q95 = struct.unpack("d", data)[0]

    log.debug('Tail latency q95: {}'.format(q95))

    return q95
