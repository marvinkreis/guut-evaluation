from string_utils.generation import roman_range

def test_roman_range():
    # Test case where both start and step are negative
    # The mutant should fail this case due to a changed condition
    result = list(roman_range(start=5, stop=1, step=-1))
    expected = ['V', 'IV', 'III', 'II', 'I']  # Expected Roman numerals
    assert result == expected, f"Expected {expected} but got {result}"
    
    # Test case where step leads to an invalid configuration
    try:
        list(roman_range(start=1, stop=5, step=-1))  # Invalid configuration
    except OverflowError:
        pass  # This is the expected behavior
    else:
        assert False, "Expected OverflowError but none was raised."

    # Additional valid case
    result = list(roman_range(stop=3, step=1))
    expected = ['I', 'II', 'III']
    assert result == expected, f"Expected {expected} but got {result}"