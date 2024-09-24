from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError

def test_strip_margin():
    # Test with valid input
    input_str = '''    line 1
    line 2
    line 3
    '''
    
    expected_output = '''line 1
line 2
line 3
'''
    
    # Test normal functionality
    assert strip_margin(input_str) == expected_output

    # Test with invalid input - should raise InvalidInputError for non-string types
    try:
        strip_margin(123)  # Invalid input (not a string)
    except InvalidInputError:
        pass  # This is expected
    else:
        raise AssertionError("Expected InvalidInputError was not raised for non-string input.")
        
    # Test empty string handling
    result = strip_margin("")  # Check for accepting empty string; it should not raise an error
    assert result == ""  # It should just return an empty string

    # Test with whitespace input handling
    result = strip_margin("     ")  # Check that it can handle whitespace correctly
    assert result == "", "Expected result for whitespace input should be an empty string."

# Note: Run this test to confirm behavior; it should pass for the correct implementation 
# and fail with the mutant due to invalid type handling logic.