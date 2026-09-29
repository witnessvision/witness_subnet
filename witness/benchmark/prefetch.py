"""One bounded, private background acquisition; never executes or scores a model."""
from __future__ import annotations
import threading
import time


class Prefetch:
    def __init__(self, entry, window, acquire, cancelled, *, timeout_s=900.):
        self.entry, self.window = dict(entry), window['id']
        self.stop_event = threading.Event()
        self.deadline = time.monotonic() + timeout_s
        self.result = self.error = None
        self.started_unix = time.time()
        self.finished_unix = None
        self.cancelled = lambda: (self.stop_event.is_set() or cancelled()
                                 or time.monotonic() >= self.deadline)
        def work():
            try:
                self.result = acquire(self.cancelled, lambda: max(0., self.deadline-time.monotonic()))
            except Exception as error:
                self.error = error
            finally:
                self.finished_unix = time.time()
        self.thread = threading.Thread(target=work, name='witness-prefetch', daemon=True)
        self.thread.start()

    def wait(self, cancelled):
        while self.thread.is_alive():
            if cancelled():
                raise InterruptedError('prefetch_wait_cancelled')
            self.thread.join(.2)
        if self.error:
            raise self.error
        return self.result

    def stop(self):
        self.stop_event.set()
