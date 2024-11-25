from codetiming import Timer


def test():
    # for similar values a-b == a%b, so we patch the timing function to start at 0 again.
    # this isn't strictly necessary to observe mutant behavior.
    import time
    start_time = time.perf_counter()
    original_function = time.perf_counter
    time.perf_counter = lambda: original_function() - start_time

    log = []

    @Timer(text="{seconds}", logger=log.append)
    def timed_fun():
        time.sleep(.1)
        pass

    timed_fun()
    seconds = float(log[0])

    assert abs(seconds - .1) < 0.01
