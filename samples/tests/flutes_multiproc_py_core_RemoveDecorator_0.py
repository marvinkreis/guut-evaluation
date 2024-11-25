from flutes.multiproc import DummyPool


def test():
    assert isinstance(DummyPool.__dict__["_no_op"], staticmethod)
