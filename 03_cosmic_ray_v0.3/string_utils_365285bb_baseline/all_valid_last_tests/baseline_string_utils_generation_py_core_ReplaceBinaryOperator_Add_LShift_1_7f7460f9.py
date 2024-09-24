from string_utils.generation import roman_range

def test__roman_range_mutant_detection():
    # Test case that checks valid inputs for forward step
    result = list(roman_range(5, start=1, step=1))
    expected = ['I', 'II', 'III', 'IV', 'V']
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case that triggers the OverflowError when the step exceeds range
    try:
        list(roman_range(5, start=6, step=1))  # start > stop
        assert False, "Expected OverflowError not raised"  # Fail if no exception
    except OverflowError:
        pass  # Correctly raised the error

    try:
        list(roman_range(5, start=1, step=5))  # step is too large
        assert False, "Expected OverflowError not raised"  # Fail if no exception
    except OverflowError:
        pass  # Correctly raised the error

    # Additional case to check the negative step
    try:
        result = list(roman_range(1, start=5, step=-1))  # valid negative step
        expected_reverse = ['V', 'IV', 'III', 'II', 'I']
        assert result == expected_reverse, f"Expected {expected_reverse}, but got {result}"
    except OverflowError:
        assert False, "Unexpected OverflowError"