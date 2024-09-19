import time
from codetiming._timer import Timer

def test__timer_functionality():
    """A change from multiplication to bitwise XOR will cause TypeErrors when calculating milliseconds."""
    timer = Timer()
    
    timer.start()
    time.sleep(0.1)  # Sleep for 100 milliseconds
    elapsed_time = timer.stop()
    
    # Verify that elapsed time is calculated as expected in seconds and milliseconds
    assert elapsed_time > 0, "Elapsed time must be greater than zero"
    assert (elapsed_time * 1000) > 0, "Milliseconds must be calculated correctly"

# Run the test to see if it correctly identifies the mutant
test__timer_functionality()