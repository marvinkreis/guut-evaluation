from string_utils.generation import roman_range

def test_roman_range():
    # Test Case 1: Valid input should yield exact Roman numerals in a known sequence
    expected_output = ['I', 'II', 'III', 'IV', 'V']
    result = list(roman_range(5, start=1, step=1))
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test Case 2: Step with values that should yield a distinct response
    expected_diverging_output = ['III', 'IV', 'V']  # from 3 to 5 should yield III, IV, V
    result_diverge = list(roman_range(5, start=3, step=1))  # <-- Closed the parenthesis here
    assert result_diverge == expected_diverging_output, f"Expected {expected_diverging_output}, but got {result_diverge}"

    # Test Case 3: Check when start=4 and step=1
    expected_step_test = ['IV', 'V']  # Should yield IV to V
    result_step = list(roman_range(5, start=4, step=1))
    assert result_step == expected_step_test, f"Expected {expected_step_test}, but got {result_step}"

    # Test Case 4: Invalid configuration that should raise an OverflowError
    try:
        # The intention is valid, but it would set limits incorrectly
        list(roman_range(2, start=3, step=1))  # 3 to 2 is invalid for a yielding case
    except OverflowError:
        pass  # Expected behavior
    else:
        assert False, "Expected an OverflowError but didn't get one"