from codetiming import Timer


def test():
    try:
        Timer(_start_time = 0.0)
        assert False
    except TypeError:
        pass
