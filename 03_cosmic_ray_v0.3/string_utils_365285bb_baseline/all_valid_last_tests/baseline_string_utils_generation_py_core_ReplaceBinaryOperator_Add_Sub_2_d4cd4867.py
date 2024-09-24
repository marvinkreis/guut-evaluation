from string_utils.generation import roman_range

def test_roman_range_mutant_detection():
    # 1. Valid backward range case should work in both the mutant and original code
    result = list(roman_range(start=5, stop=1, step=-1))  # Should yield: ['V', 'IV', 'III', 'II', 'I']
    expected_result = ['V', 'IV', 'III', 'II', 'I']
    assert result == expected_result, f"Expected {expected_result}, got {result}"

    # 2. Now we check a case that should trigger the incorrect logic in the mutant
    try:
        # This case should be valid in original but invalid for the mutant.
        list(roman_range(start=1, stop=3, step=-1))  # Expected to raise OverflowError in mutant
        assert False, "Expected OverflowError not raised"  # This line should not be reached
    except OverflowError as e:
        # Correctly raised error for mutant due to incorrect logic
        assert str(e) == 'Invalid start/stop/step configuration', "Unexpected error message"
    except Exception as e:
        assert False, f"Expected OverflowError, but got a different exception: {e}"

# Run the test function
test_roman_range_mutant_detection()