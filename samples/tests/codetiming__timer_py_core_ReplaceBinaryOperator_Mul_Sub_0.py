from codetiming import Timer
import time


def test():
    log = []

    @Timer(text="{seconds},{milliseconds}", logger=log.append)
    def timed_fun():
        time.sleep(.001)
        pass

    timed_fun()
    seconds = float(log[0].split(",")[0])
    milliseconds = float(log[0].split(",")[1])

    assert abs((seconds * 1000) - milliseconds) < 0.001
