from possible_change import possible_change

def test__possible_change():
    """The mutant fails because it does not handle the case when there are no coins."""
    assert possible_change([], 1) == 0, "Should return 0 ways to change for non-zero total"
    assert possible_change([], 5) == 0, "Should return 0 ways to change for non-zero total"
    assert possible_change([], 10) == 0, "Should return 0 ways to change for non-zero total"
    assert possible_change([], 0) == 1, "Should return 1 way to change for total of 0"

test__possible_change()