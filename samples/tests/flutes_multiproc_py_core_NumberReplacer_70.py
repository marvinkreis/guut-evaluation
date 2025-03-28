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
        assert n == 1

    p.update = assert_update
    list(p.new([1,2,3,4]))
