# Run: `python -m awe.inference`

import os
import select
import time
from typing import Callable

# IMPORTANT: Keep consistent with constants in `js/ipc.ts`.
INCOMING_PATH = 'js_to_awe'
OUTGOING_PATH = 'awe_to_js'

class Communicator:
    """Communicates via IPC with TypeScript module `js/ipc.ts`."""

    outgoint_pipe: int

    def __init__(self, handler: Callable[[int]] = lambda _: print('received a message')):
        self.handler = handler

    def run(self):
        os.mkfifo(INCOMING_PATH)

        try:
            incoming_pipe = os.open(INCOMING_PATH, os.O_RDONLY | os.O_NONBLOCK)
            print('incoming pipe ready')

            # Open outgoing pipe.
            while True:
                try:
                    self.outgoint_pipe = os.open(OUTGOING_PATH, os.O_WRONLY)
                    print('outgoing pipe ready')
                    break
                except IOError:
                    # Wait until initialized.
                    time.sleep(1)
                    continue

            try:
                poll = select.poll()
                poll.register(incoming_pipe, select.POLLIN)

                try:
                    while True:
                        # Read every 1 second.
                        if (incoming_pipe, select.POLLIN) in poll.poll(1000):
                            self.handler(incoming_pipe)
                finally:
                    poll.unregister(incoming_pipe)
            finally:
                os.close(incoming_pipe)
        finally:
            os.remove(INCOMING_PATH)
            os.remove(OUTGOING_PATH)
