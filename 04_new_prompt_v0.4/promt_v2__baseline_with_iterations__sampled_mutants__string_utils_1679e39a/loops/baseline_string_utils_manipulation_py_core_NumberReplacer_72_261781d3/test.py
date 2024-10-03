from string_utils.manipulation import prettify

def test__prettify():
    """
    Test whether spaces are correctly formatted around punctuation by checking a specific case. 
    The input string has inconsistent spacing: 'hello !  how are you ?'. 
    The mutant improperly accesses the second capture group instead of the intended first group for spacing adjustment,
    which will result in incorrect formatting and will produce a different output than the correct implementation.
    """
    output = prettify('hello !  how are you ?')
    assert output == 'Hello! How are you?'