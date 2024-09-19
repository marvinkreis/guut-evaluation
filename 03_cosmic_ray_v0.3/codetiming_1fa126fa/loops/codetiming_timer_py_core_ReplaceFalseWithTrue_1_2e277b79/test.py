from codetiming._timer import Timer

def test__timer_repr():
    """The mutant Timer exposes _start_time in its representation, which can lead to unintended information leakage."""
    timer = Timer(name="Test Timer")
    timer.start()
    repr_output = repr(timer)
    
    # The test will fail if _start_time is seen in the representation
    assert "_start_time=" not in repr_output, "Timer representation should not expose _start_time"