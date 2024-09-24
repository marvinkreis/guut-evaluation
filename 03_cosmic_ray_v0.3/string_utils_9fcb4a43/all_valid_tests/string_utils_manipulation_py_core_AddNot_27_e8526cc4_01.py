from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError  

# Mock implementation of the mutant strip_margin function
def mutant_strip_margin(input_string):
    # This version incorrectly allows non-string inputs to pass without raising an error
    return ' '.join(input_string.split('\n')) if isinstance(input_string, str) else input_string  # Incorrectly handles input

def test__strip_margin():
    """The mutant version of strip_margin fails to raise an InvalidInputError when given non-string inputs."""
    
    # Testing with non-string inputs
    non_string_inputs = [None, 123]

    for input_data in non_string_inputs:
        # Testing the correct implementation
        try:
            strip_margin(input_data)  # This should raise an error.
            raise AssertionError("Expected InvalidInputError not raised for input: {}".format(input_data))
        except InvalidInputError:
            pass  # This is expected behavior for the correct implementation.

        # Testing the mutant implementation
        result = mutant_strip_margin(input_data)  # This call should allow non-string inputs.
        if result is not input_data:
            raise AssertionError("Mutant misbehaved for input: {}. Got result: {}".format(input_data, result))

# Running the test function
test__strip_margin()