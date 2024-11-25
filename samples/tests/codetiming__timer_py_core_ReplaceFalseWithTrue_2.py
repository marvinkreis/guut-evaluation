from codetiming import Timer


def test():
    try:
        Timer(last = 0.0)
        assert False
    except TypeError:
        pass
