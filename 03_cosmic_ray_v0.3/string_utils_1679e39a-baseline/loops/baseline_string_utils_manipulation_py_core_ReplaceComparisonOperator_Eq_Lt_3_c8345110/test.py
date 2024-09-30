from string_utils.manipulation import compress

def test__compress():
    """
    This test checks if the compress function properly raises a ValueError when given an empty string.
    The mutant changes the condition to check for negative length of the string, which can never be true.
    Therefore, it bypasses the error check for empty input strings, leading to incorrect behavior.
    This test will fail on the mutant because the mutant does not raise an error when an empty string is passed.
    """
    try:
        compress('')
    except ValueError as e:
        assert str(e) == 'Input string cannot be empty'
    else:
        assert False, "Expected ValueError not raised"