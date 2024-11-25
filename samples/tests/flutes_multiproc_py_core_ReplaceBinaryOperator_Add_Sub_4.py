from flutes.multiproc import StatefulPool


class MockPool:
    def __init__(self, *args, **kwargs):
        assert len(args) == 4

    def __getattribute__(self, name):
        return None

class MockState:
    def __init__(self, arg, *args, **kwargs):
        pass


def test():
    pool = StatefulPool(MockPool, MockState, (), (1,2,3,4), {})
