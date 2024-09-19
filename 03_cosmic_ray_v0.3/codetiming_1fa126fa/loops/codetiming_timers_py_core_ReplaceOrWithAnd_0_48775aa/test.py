from codetiming._timers import Timers

def test__timers_min():
    """The mutant changes min function to return 0 instead of the correct minimum when negative values are present."""
    timers = Timers()
    timers.add("test_timer", 3)
    timers.add("test_timer", -1)
    timers.add("test_timer", 0)
    
    # The correct min should be -1
    output = timers.min("test_timer")
    assert output == -1, "min should return the minimum value, not 0"