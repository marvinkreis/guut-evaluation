import time
from codetiming._timer import Timer

def test__timer_milliseconds():
    """Test that ensures the mutant alters the milliseconds calculation to yield lower values."""
    with Timer() as timer:
        time.sleep(1)  # Allow the timer to run for about 1 second
    
    elapsed_ms = timer.last * 1000  # Calculate elapsed time in milliseconds
    assert elapsed_ms > 1000, "Timer must report elapsed time greater than 1000 ms"
    
    # Check that the mutant's adjustment will yield a value below expected
    assert elapsed_ms < 1000.5, "Mutant's reported milliseconds should be less than 1000.5 ms"