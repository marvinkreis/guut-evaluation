from string_utils.validation import is_full_string

def test__is_full_string():
    # Testing valid full string
    assert is_full_string("hello") == True

    # Testing empty string
    assert is_full_string("") == False

    # Testing string with only spaces
    assert is_full_string("   ") == False

    # Testing None input
    assert is_full_string(None) == False

    # Testing a string with spaces that is not just white space
    assert is_full_string(" hello ") == True

    # This case is specifically to detect the mutant:
    # In the mutant, a string that only contains spaces or is empty would incorrectly return True
    assert is_full_string(" ") == False