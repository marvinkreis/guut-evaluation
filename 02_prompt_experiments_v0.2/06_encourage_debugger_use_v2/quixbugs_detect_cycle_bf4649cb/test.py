from detect_cycle import detect_cycle

def test__detect_cycle():
    """The mutant fails when detecting cycles in an empty linked list."""
    output = detect_cycle(None)
    assert output == False, "detect_cycle must return False for an empty linked list"