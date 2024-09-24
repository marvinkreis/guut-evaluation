from string_utils.validation import is_full_string

def test__is_full_string():
    # Test cases to validate the original behavior
    assert is_full_string('hello') == True  # 'hello' is a full string
    assert is_full_string(' ') == False      # ' ' is not considered a full string
    assert is_full_string('   ') == False    # '   ' is not considered a full string
    assert is_full_string('foo bar') == True  # 'foo bar' is a full string
    assert is_full_string('') == False        # '' is not a full string