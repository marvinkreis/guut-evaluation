from string_utils.manipulation import prettify

def test__prettify_capitalization():
    """
    Test the behavior of prettify on an input string where a period is followed by a lowercase letter.
    This verifies that the first letter after a period is capitalized correctly.
    The mutant should produce an incorrect output because it alters the logic of capitalization.
    """
    output = prettify('hello. this should be capitalized.')
    assert output == 'Hello. This should be capitalized.'