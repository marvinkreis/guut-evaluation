from string_utils.generation import roman_range

def test_roman_range():
    # Test normal operation
    result = list(roman_range(5, start=1, step=1))  # Should return: ['I', 'II', 'III', 'IV', 'V']
    assert result == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {result}"

    # Test reverse range
    result_reverse = list(roman_range(1, start=5, step=-1))  # Should return: ['V', 'IV', 'III', 'II', 'I']
    assert result_reverse == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {result_reverse}"

    # Test with invalid configuration that should raise OverflowError in the mutant due to start>stop
    try:
        # This should yield an OverflowError in the mutant code because start > stop while step is positive
        list(roman_range(1, start=3, step=1))  # Invalid case in the mutant should trigger error
        assert False, "Expected OverflowError from mutant due to invalid configuration"
    except OverflowError:
        pass  # Expected outcome

    # Test valid step conditions which are also valid inputs
    valid_result = list(roman_range(4, start=1, step=1))  # Should return ['I', 'II', 'III', 'IV']
    assert valid_result == ['I', 'II', 'III', 'IV'], f"Expected ['I', 'II', 'III', 'IV'], got {valid_result}"

# Execute the test
test_roman_range()