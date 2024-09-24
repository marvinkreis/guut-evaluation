from string_utils.generation import roman_range

def test_roman_range():
    # Test when the parameters are valid and should generate proper roman numerals
    expected_output = ['I', 'II', 'III', 'IV', 'V']
    result = list(roman_range(5))
    assert result == expected_output, f"Expected {expected_output} but got {result}"

    # Test invalid configuration that should raise OverflowError
    try:
        list(roman_range(1, start=5, step=1))  # This should raise an OverflowError
        assert False, "Expected OverflowError not raised"
    except OverflowError:
        pass  # Correct behavior, passed the test

    # Test with invalid step (negative step from higher to lower)
    try:
        list(roman_range(5, start=1, step=-1))  # This should raise an OverflowError
        assert False, "Expected OverflowError not raised"
    except OverflowError:
        pass  # Correct behavior, passed the test