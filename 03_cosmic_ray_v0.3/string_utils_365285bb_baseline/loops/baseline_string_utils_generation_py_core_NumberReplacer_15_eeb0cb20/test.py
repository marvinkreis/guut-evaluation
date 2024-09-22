from string_utils.generation import roman_range

def test_roman_range():
    # Test case where step is negative and should generate a decreasing sequence
    result = list(roman_range(1, start=7, step=-1))
    expected = ['VII', 'VI', 'V', 'IV', 'III', 'II', 'I']
    
    # This assertion should pass with the correct implementation
    assert result == expected, f"Expected: {expected}, but got: {result}"

    # Test case where step is -1 but stop is greater than start
    try:
        list(roman_range(10, start=5, step=-1))
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration', "Expected an OverflowError"

    # Test case where step is -2 and the function should handle this and generate a valid range
    result = list(roman_range(1, start=7, step=-2))
    expected = ['VII', 'V', 'III', 'I']
    
    # This assertion should pass with the correct implementation
    assert result == expected, f"Expected: {expected}, but got: {result}"

# You can add the below line to call the test
test_roman_range()