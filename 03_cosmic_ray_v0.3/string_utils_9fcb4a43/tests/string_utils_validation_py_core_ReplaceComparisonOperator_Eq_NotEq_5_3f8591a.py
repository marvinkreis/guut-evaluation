from string_utils.validation import is_isogram

def test__is_isogram():
    """The mutant version of is_isogram incorrectly returns True for non-isogram strings."""
    assert is_isogram("hello") == False, "is_isogram should return False for non-isograms"
    assert is_isogram("abcde") == True, "is_isogram should return True for isograms"