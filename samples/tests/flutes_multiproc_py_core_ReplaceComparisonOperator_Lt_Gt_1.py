import multiprocessing as mp
from types import SimpleNamespace
from time import sleep
from threading import Thread

from flutes.multiproc import StatefulPool
import flutes.multiproc as m
import flutes


class MockState(flutes.PoolState):
    def simple_fn(self, x):
        pass

def test():
    with flutes.safe_pool(processes=4, state_class=MockState) as pool_stateful:
        try:
            pool_stateful.broadcast(MockState.simple_fn)
            assert False
        except TypeError:
            pass

