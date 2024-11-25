from flutes.multiproc import ProgressBarManager
from multiprocessing import Queue


class MockInt:
    def __eq__(self, other):
        return True

class MockQueue:
    def __init__(self):
        self.events = []

    def put_nowait(self, event):
        self.events.append(event)


def test():
    Proxy = ProgressBarManager.Proxy
    queue = MockQueue()
    p = Proxy(queue)

    l = [1, 2, 3, 4]
    _enumerate = __builtins__["enumerate"]
    __builtins__["enumerate"] = lambda x: [(MockInt(), y) for y in x] if x is l else _enumerate(x)
    try:
        list(p._iter_per_percentage(l, 4, .1))
        assert False
    except TypeError:
        pass
    finally:
        __builtins__["enumerate"] = enumerate
