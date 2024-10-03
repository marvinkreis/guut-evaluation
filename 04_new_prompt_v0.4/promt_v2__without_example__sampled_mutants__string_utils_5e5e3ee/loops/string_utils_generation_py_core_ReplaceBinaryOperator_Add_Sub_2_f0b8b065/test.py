from string_utils.generation import roman_range

def test__roman_range_no_steps():
    """
    Test the behavior of roman_range where start equals stop with a negative step (start=5, stop=5, step=-1).
    The mutant should raise an OverflowError due to the modified validation logic,
    while the baseline should handle it gracefully without raising an error.
    """
    try:
        for _ in roman_range(stop=5, start=5, step=-1):
            pass
        assert False, "Expected OverflowError was not raised"
    except OverflowError:
        assert True