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

    def assert_update(n):
        assert n == 4
    p.update = assert_update

    print(list(p._iter_per_percentage([], 4, .1)))
