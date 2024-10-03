from string_utils.manipulation import compress

def test__compress():
    """
    Test whether the compress function raises a ValueError for an invalid compression level
    (specifically, a level above 9). The input 'Hello World' with a compression level of 10 
    should trigger this error. The mutant erroneously changes the condition to allow levels below 9,
    which would not trigger the error in this case and would cause the test to pass incorrectly.
    """
    try:
        compress('Hello World', compression_level=10)
        assert False, "Expected ValueError was not raised."
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'