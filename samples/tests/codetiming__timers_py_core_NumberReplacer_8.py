from codetiming._timers import Timers


def test():
    a = Timers()
    a.data[""] = []
    a._timings[""] = []
    assert a.median("") == 0
