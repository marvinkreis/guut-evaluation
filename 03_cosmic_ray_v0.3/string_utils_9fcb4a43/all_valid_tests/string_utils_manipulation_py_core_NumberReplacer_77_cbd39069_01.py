from string_utils.manipulation import prettify

def test__prettify():
    """The mutant doesn't normalize spacing correctly due to changes in __ensure_spaces_around."""
    output = prettify('Hello     world!   How are you?  ')
    assert output == 'Hello world! How are you?', "prettify must return normalized string"

    # Test with multiple spaces in a different formation
    output2 = prettify('This is  a test.   With irregular spacing!  ')
    assert output2 == 'This is a test. With irregular spacing!', "prettify must normalize additional spaces"