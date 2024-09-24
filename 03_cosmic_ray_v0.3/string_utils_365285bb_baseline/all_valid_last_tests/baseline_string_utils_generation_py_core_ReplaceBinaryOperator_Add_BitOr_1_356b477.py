from string_utils.generation import roman_range

def test_roman_range():
    # Basic functionality test
    result = list(roman_range(3))  # This should yield ['I', 'II', 'III']
    expected_output = ['I', 'II', 'III']
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test another valid output
    result2 = list(roman_range(4))  # Expected: ['I', 'II', 'III', 'IV']
    expected_output2 = ['I', 'II', 'III', 'IV']
    assert result2 == expected_output2, f"Expected {expected_output2}, but got {result2}"

    # Test condition that will help catch the mutant's logical misalignment
    result_mutant_check = list(roman_range(5, start=1, step=2))  # Expected: ['I', 'III', 'V']
    expected_output_mutant_check = ['I', 'III', 'V']
    
    # Assert the output and force detection
    assert result_mutant_check == expected_output_mutant_check, f"Expected {expected_output_mutant_check}, but got {result_mutant_check}"

    # Check upper boundary with step leading to missed items
    result_boundary = list(roman_range(6, start=2, step=2))  # Expected: ['II', 'IV', 'VI']
    expected_output_boundary = ['II', 'IV', 'VI']
    assert result_boundary == expected_output_boundary, f"Expected {expected_output_boundary}, but got {result_boundary}"

    # Test scenario that should cause failure in mutant logic
    try:
        result_invalid = list(roman_range(4, start=5, step=1))  # Invalid range; should technically raise OverflowError
        # Since we expect the mutant to behave incorrectly, we'll note this result instead of asserting False
        assert False, "Expected OverflowError but none was raised."
    except OverflowError:
        pass  # This indicates the correct implementation functionality
