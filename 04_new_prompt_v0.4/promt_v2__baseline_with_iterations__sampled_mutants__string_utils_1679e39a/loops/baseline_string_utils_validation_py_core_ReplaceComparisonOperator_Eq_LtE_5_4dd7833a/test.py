from string_utils.validation import is_isogram

def test__is_isogram():
    """
    Test whether a string is correctly identified as an isogram. The input 'abcde' is an isogram since all
    characters are unique, while 'aabbcc' is not an isogram because the characters repeat. The mutant logic
    incorrectly allows repeated characters by using `<=` instead of `==`.
    """
    output = is_isogram('abcde')
    assert output == True
    output_mutant = is_isogram('aabbcc')
    assert output_mutant == False