import sys
from flake8.utils import parameters_for
import inspect


def fun(a, b):
    pass

class MockPlugin:
    plugin = fun

def test():
    sys.version_info = (3, 2)

    try:
        parameters_for(MockPlugin)
    except ZeroDivisionError:
        assert False
