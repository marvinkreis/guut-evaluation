from string_utils.manipulation import compress

def test__compress():
    """
    Test whether the compress function raises a ValueError when an invalid compression level (non-integer) is provided.
    The input 'test' with a compression level of 'invalid' will raise an error if the original code is intact, 
    but will pass silently if the mutant is present since the condition for checking the type of compression_level is faulty.
    """
    try:
        compress("test", compression_level='invalid')
        assert False, "ValueError was not raised."
    except ValueError:
        pass  # This is expected, so the test passes