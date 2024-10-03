from string_utils.manipulation import compress

def test__compress():
    """
    Test that the compress function raises a ValueError when provided with an invalid compression level of -1.
    The original implementation only allows integers between 0 and 9 (inclusive), but the mutant incorrectly 
    allows -1 as a valid input, leading to different behavior when handling invalid compression levels.
    """
    try:
        compress("example string", compression_level=-1)
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'
    else:
        raise AssertionError("ValueError was not raised for an invalid compression_level")