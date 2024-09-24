from string_utils.generation import roman_range

def test_roman_range():
    # TEST CASE 1: Invalid case, start equals stop with a negative step
    try:
        # This should raise an OverflowError in the original implementation
        list(roman_range(start=4, stop=4, step=-1))
    except OverflowError:
        # Correct behavior for the original code
        pass
    else:
        raise AssertionError("Expected OverflowError not raised for start == stop with negative step.")

    # TEST CASE 2: Valid case with a forward range
    result = list(roman_range(start=1, stop=5, step=1))  # Expecting ['I', 'II', 'III', 'IV', 'V']
    expected_result = ['I', 'II', 'III', 'IV', 'V']
    assert result == expected_result, f"Expected {expected_result} but got {result}"

    # TEST CASE 3: Valid case with a backward range
    result_backwards = list(roman_range(start=5, stop=1, step=-1))  # Expecting ['V', 'IV', 'III', 'II', 'I']
    expected_result_backwards = ['V', 'IV', 'III', 'II', 'I']
    assert result_backwards == expected_result_backwards, f"Expected {expected_result_backwards} but got {result_backwards}"

    # TEST CASE 4: Check invalid condition again for a different edge case
    try:
        # This should raise OverflowError since start == stop for negative step
        list(roman_range(start=3, stop=3, step=-1))
    except OverflowError:
        # Correct behavior
        pass
    else:
        raise AssertionError("Expected OverflowError not raised for start == stop with negative step.")