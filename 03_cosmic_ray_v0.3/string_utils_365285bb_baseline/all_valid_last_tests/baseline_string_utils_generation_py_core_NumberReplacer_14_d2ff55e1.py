from string_utils.generation import roman_range

def test_valid_backward_range():
    # Valid backward range from 5 to 1
    expected_output = ['V', 'IV', 'III', 'II', 'I']
    result = list(roman_range(start=5, stop=1, step=-1))
    
    # This should work in both the original and mutant editions
    assert result == expected_output, f"Expected {expected_output} but got {result}"

def test_invalid_backward_range():
    # Invalid configuration: this should raise an OverflowError
    try:
        list(roman_range(start=1, stop=5, step=-1))  # Invalid backward traversal
        raise AssertionError("Expected OverflowError but none was raised.")
    except OverflowError:
        # Expecting the error; this confirms mutant fails on an invalid backward attempt.
        pass

def test_invalid_forward_range():
    # Trying to go from 5 to 1 with a positive step should raise an error
    try:
        list(roman_range(start=5, stop=1, step=1))  # Invalid forward configuration
        raise AssertionError("Expected OverflowError but none was raised.")
    except OverflowError:
        # Expected behavior confirming error handling in both implementations
        pass

def test_valid_backward_from_three_to_one():
    # Valid backward movement from 3 to 1
    expected_output = ['III', 'II', 'I']
    result = list(roman_range(start=3, stop=1, step=-1))

    assert result == expected_output, f"Expected {expected_output} but got {result}"

# Execute tests
test_valid_backward_range()                 # Should pass: valid range from 5 to 1
test_invalid_backward_range()               # Should trigger the mutant's logic error
test_invalid_forward_range()                # Should confirm invalid forward configuration handling
test_valid_backward_from_three_to_one()    # Should pass: valid range from 3 to 1