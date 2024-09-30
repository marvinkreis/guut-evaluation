from string_utils.validation import is_pangram

def test__is_pangram():
    """
    Test whether a string that contains all letters of the alphabet is correctly identified as a pangram.
    The input 'The quick brown fox jumps over the lazy dog' contains all letters from a to z, so it should return true.
    The mutant incorrectly negates the full string check, which would cause it to return false for valid pangrams.
    """
    output = is_pangram('The quick brown fox jumps over the lazy dog')
    assert output == True