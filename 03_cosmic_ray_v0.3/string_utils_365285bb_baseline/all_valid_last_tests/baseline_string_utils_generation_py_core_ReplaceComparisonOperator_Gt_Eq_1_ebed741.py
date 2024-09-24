from string_utils.generation import roman_range

def test_roman_range_invalid_configuration():
    try:
        # Calling roman_range with start greater than stop and step > 0
        list(roman_range(5, start=10, step=1))
        # If no error is raised, the test fails
        assert False, "Expected OverflowError not raised."
    except OverflowError:
        # This exception is expected; if we reach this point the test passes
        pass