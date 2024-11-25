import multiprocessing as mp
from types import SimpleNamespace
from collections import UserString

from flutes.multiproc import StatefulPool

class MockString(UserString):
    pass

class MockPool:
    def __init__(self, arg, *args, **kwargs):
        pass
    def __getattribute__(self, name):
        return None

class MockState:
    def __init__(self, arg, *args, **kwargs):
        pass


def test():
    pool = StatefulPool(MockPool, MockState, (), (1,2), {})
    pool._pool = SimpleNamespace(_state=MockString(mp.pool.RUN))

    try:
        pool.broadcast(lambda x: 1)
    except ValueError as e:
        assert "Only unbound" in str(e)
