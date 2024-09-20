from string_utils.manipulation import booleanize

def test__booleanize():
    """The mutant changes the input validation logic which should not trigger an error for valid strings."""
    output = booleanize('true')
    assert output is True, "booleanize should return True for input 'true'"