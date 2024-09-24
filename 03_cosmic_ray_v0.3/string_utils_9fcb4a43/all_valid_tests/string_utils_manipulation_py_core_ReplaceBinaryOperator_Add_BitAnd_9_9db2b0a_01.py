from string_utils.manipulation import prettify

def test__prettify():
    """The mutant uses '&' instead of '+', which would cause incorrect string formatting."""
    input_string = '   This is a test string.  '
    correct_output = prettify(input_string)
    assert correct_output == 'This is a test string.', "prettify should format the string correctly."