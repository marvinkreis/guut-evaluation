# Mock implementation of is_string function
def is_string(input_value):
    return isinstance(input_value, str)

# Custom exception for invalid input
class InvalidInputError(Exception):
    pass

# Original implementation of the booleanize function
def booleanize(input_string) -> bool:
    if not is_string(input_string):  # Correct implementation that raises on invalid input
        raise InvalidInputError(input_string)
    
    return input_string.lower() in ('true', '1', 'yes', 'y')

# Mutant version of the booleanize function
def mutated_booleanize(input_string) -> bool:
    if is_string(input_string):
        return input_string.lower() in ('true', '1', 'yes', 'y')
    
    # Returning False for non-strings, allowing them through without raising errors
    return False  

def test_booleanize():
    # Testing valid inputs for the original implementation
    assert booleanize('true') == True
    assert booleanize('1') == True
    assert booleanize('yes') == True
    assert booleanize('y') == True
    assert booleanize('nope') == False
    assert booleanize('false') == False
    assert booleanize('0') == False

    # Test the original implementation for invalid inputs
    invalid_inputs = [123, None, [], {}, True, 3.14]  # A variety of non-string inputs

    # Check that the original raises an exception for each invalid input
    for input_value in invalid_inputs:
        try:
            booleanize(input_value)  # This should raise InvalidInputError
            assert False, f"Expected InvalidInputError for input: {input_value}"
        except InvalidInputError:
            pass  # Expected behavior

    # Testing the mutant version with valid inputs
    assert mutated_booleanize('true') == True
    assert mutated_booleanize('1') == True
    assert mutated_booleanize('yes') == True
    assert mutated_booleanize('y') == True
    assert mutated_booleanize('nope') == False

    # Test the mutant function with invalid inputs
    for input_value in invalid_inputs:
        result = mutated_booleanize(input_value)  # The mutant should not raise and return False
        assert result == False, f"Expected False for input: {input_value} in mutated version"

    print("All tests passed!")  # Final confirmation that all tests succeeded

# Run the test
test_booleanize()