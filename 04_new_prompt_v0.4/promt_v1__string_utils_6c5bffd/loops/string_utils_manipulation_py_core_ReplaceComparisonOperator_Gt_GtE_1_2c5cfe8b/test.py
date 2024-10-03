from string_utils.manipulation import compress
from string_utils.errors import InvalidInputError  # Import necessary error class

def test__compress_large_input():
    """
    Test the compress function with a very long string and a compression level of 9. 
    The baseline should succeed and return a compressed string, while the mutant should
    raise a ValueError due to the mutant's flawed compression level validation.
    This test checks that the output from the baseline is not equal to the expected raised error.
    """
    long_string = "Lorem ipsum dolor sit amet, " * 100  # Create a long input string
    try:
        output = compress(long_string, compression_level=9)
        print(f"output = {output}")
        assert isinstance(output, str)  # Assert that the output is a string
        assert len(output) > 0  # Assert that the output is not empty
    except ValueError as e:
        print(f"ValueError: {e}")
        assert False, "Expected successful compression, but raised a ValueError."