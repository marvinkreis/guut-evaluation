from string_utils.generation import roman_range

def test__roman_range():
    """
    Test that the roman_range function correctly raises an OverflowError for a configuration 
    where the range is impossible (e.g., start = 1, stop = 1, step = -1). The baseline should 
    raise an OverflowError here, while the mutant may not handle this case correctly due to its logic change.
    """
    # Invalid configuration: start equals stop but with step -1
    try:
        list(roman_range(start=1, stop=1, step=-1))  # This should raise an OverflowError
        assert False, "Expected OverflowError for invalid input (start=1, stop=1, step=-1), but no error was raised."
    except OverflowError:
        pass  # This is the expected outcome, that means the test passes for baseline

    # Valid configuration should return proper Roman numeral sequence
    valid_result = list(roman_range(start=1, stop=5, step=1))  # Valid range
    assert valid_result == ['I', 'II', 'III', 'IV', 'V'], "Expected Roman numerals from I to V"