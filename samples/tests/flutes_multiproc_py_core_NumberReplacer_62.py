from flutes.multiproc import ProgressBarManager
from multiprocessing import Queue


class MockQueue:
    def __init__(self):
        self.events = []

    def put_nowait(self, event):
        self.events.append(event)


def test():
    Proxy = ProgressBarManager.Proxy
    queue = MockQueue()
    p = Proxy(queue)

    def assert_update_frequency(iterable, update_frequency):
        assert update_frequency == 1

    p._iter_per_elems = assert_update_frequency
    p.new([])
