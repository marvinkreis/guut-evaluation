from string_utils.manipulation import prettify

def test__prettify():
    """The mutant's change in the __ensure_spaces_around function may lead to extra spaces at the end of a string."""
    input_string = "Hello,    world!  This is a test. "
    output = prettify(input_string)
    assert output == 'Hello, world! This is a test.', "prettify should format the string correctly without trailing spaces."