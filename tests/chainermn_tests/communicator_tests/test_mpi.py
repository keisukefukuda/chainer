import os
try:
    import queue
except ImportError:
    import Queue as queue
import re
import signal
from subprocess import CalledProcessError
from subprocess import PIPE
from subprocess import Popen
import sys
import threading
import unittest

import numpy as np
import pytest

import chainer.testing
import chainer.testing.attr
import chainermn
from chainermn.communicators import _memory_utility
from chainermn.testing.device import get_device
import chainerx as chx


class _TimeoutThread(threading.Thread):
    def __init__(self, queue, rank):
        super(_TimeoutThread, self).__init__()
        self.queue = queue
        self.rank = rank

    def run(self):
        try:
            self.queue.get(timeout=60)
        except queue.Empty:
            # Show error message and information of the problem
            try:
                p = Popen(['ompi_info', '--all', '--parsable'], stdout=PIPE)
                out, err = p.communicate()
                if type(out) == bytes:
                    out = out.decode('utf-8')
                m = re.search(r'ompi:version:full:(\S+)', out)
                version = m.group(1)

                msg = "\n\n" \
                      "***** ERROR: bcast test deadlocked. " \
                      "One of the processes " \
                      "***** crashed or you encountered a known bug of " \
                      "Open MPI." \
                      "***** The following Open MPI versions have a bug \n" \
                      "***** that cause MPI_Bcast() deadlock " \
                      "when GPUDirect is used: \n" \
                      "***** 3.0.0, 3.0.1, 3.0.2, 3.1.0, 3.1.1, 3.1.2\n" \
                      "***** Your Open MPI version: {}\n".format(version)
                if self.rank == 1:
                    # Rank 1 prints the error message.
                    # This is because rank 0 is the root of Bcast(), and it
                    # may finish Bcast() immediately
                    # without deadlock, depending on the timing.
                    print(msg.format(version))
                    sys.stdout.flush()

                os.kill(os.getpid(), signal.SIGKILL)
            except CalledProcessError:
                pass


@pytest.mark.parametrize('use_chainerx', [True, False])
def test_bcast_deadlock(use_chainerx):
    comm = chainermn.create_communicator('flat')
    mpi_comm = comm.mpi_comm
    buf_size = 100

    device_id = comm.intra_rank
    device = get_device(device_id, use_chainerx)
    # device.use()
    chainer.cuda.get_device_from_id(device_id).use()

    if comm.size < 2:
        pytest.skip('This test is for at least two processes')

    q = queue.Queue(maxsize=1)

    if use_chainerx:
        # chainerx
        if comm.rank == 0:
            array = chx.arange(buf_size, dtype=chx.float32, device=device.device)
        else:
            array = chx.empty(buf_size, dtype=chx.float32, device=device.device)
    else:
        # numpy/cupy
        if comm.rank == 0:
            array = np.arange(buf_size, dtype=np.float32)
        else:
            array = np.empty(buf_size, dtype=np.float32)

        array = chainer.cuda.to_gpu(array, device=device_id)

    ptr = _memory_utility.array_to_buffer_object(array)

    # This Bcast() cause deadlock if the underlying MPI has the bug.
    th = _TimeoutThread(q, comm.rank)
    th.start()
    mpi_comm.Bcast(ptr, root=0)
    mpi_comm.barrier()
    q.put(True)
    assert True
