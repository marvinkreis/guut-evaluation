from codetiming._timers import Timers

def test__max():
    """The mutant will fail this test by returning 0 instead of the correct max value."""
    timers = Timers()
    timers.add("test_timer", 2.5)  # Add a timing
    output = timers.max("test_timer")
    assert output == 2.5, "The max value must return the actual max timing."