from string_utils.manipulation import compress

def test__compress_negative_level():
    """
    Test the compress function with compression_level set to -1.
    The baseline should raise a ValueError, while the mutant should not.
    This test checks that valid compression levels are enforced in the baseline.
    """
    try:
        compress("This is a test string.", compression_level=-1)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'