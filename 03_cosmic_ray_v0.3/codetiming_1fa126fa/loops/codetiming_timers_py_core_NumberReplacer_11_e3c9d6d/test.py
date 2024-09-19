from codetiming._timers import Timers
import math

def test__stdev():
    """The mutant incorrectly allows the calculation of stdev with a single timing, leading to an error."""
    timers = Timers()
    timers.add('test_timer', 10.0)
    
    # The correct code should return math.nan
    try:
        result = timers.stdev('test_timer')
        assert result is math.nan, "stdev should return nan for a single timing"
    except Exception:
        assert False, "stdev should not raise an error for a single timing"
