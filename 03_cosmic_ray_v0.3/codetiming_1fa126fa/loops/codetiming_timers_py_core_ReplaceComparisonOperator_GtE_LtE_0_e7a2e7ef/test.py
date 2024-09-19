import math
from codetiming._timers import Timers

def test__timers_stdev():
    """The mutant implementation incorrectly returns nan for valid stdev calculations."""
    timers = Timers()
    timers.add('test_timer', 1.0)
    timers.add('test_timer', 2.0)
    timers.add('test_timer', 3.0)
    
    correct_result = timers.stdev('test_timer')
    assert correct_result == 1.0, "Expected standard deviation to be 1.0"
    assert not math.isnan(correct_result), "Standard deviation must not be nan"