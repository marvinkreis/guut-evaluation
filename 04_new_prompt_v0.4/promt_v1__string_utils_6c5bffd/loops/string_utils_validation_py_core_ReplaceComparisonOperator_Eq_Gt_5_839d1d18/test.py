from string_utils.validation import is_isogram

def test__is_isogram_kill_mutant():
    """
    This test checks the is_isogram function with known isogram and non-isogram strings.
    The string 'dermatoglyphics' should return True, while 'hello' should return False.
    The mutant will fail this test while the baseline will pass it.
    """
    assert is_isogram('dermatoglyphics') == True, "Expected True for isogram 'dermatoglyphics'"
    assert is_isogram('hello') == False, "Expected False for non-isogram 'hello'"