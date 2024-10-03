from string_utils.generation import random_string

def test__random_string():
    """
    Test that a randomly generated string of a valid size is not empty and has the expected length.
    The input size is 10, and the expected output length is 10. 
    The mutant's change to character generation will produce either an empty string 
    or raise an error, which will cause the assertion to fail.
    """
    output = random_string(10)
    assert len(output) == 10  # Ensures that the output string has the correct length of 10.