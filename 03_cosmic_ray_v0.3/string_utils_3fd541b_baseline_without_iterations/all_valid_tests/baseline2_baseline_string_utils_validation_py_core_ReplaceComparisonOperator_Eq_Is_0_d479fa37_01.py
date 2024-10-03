from string_utils.validation import is_isogram

def test__is_isogram():
    # Test cases for is_isogram function
    assert is_isogram('abcd') == True  # All unique characters
    assert is_isogram('hello') == False  # 'l' is repeated
    assert is_isogram('world') == True  # All unique characters
    assert is_isogram('abcdefg') == True  # All unique characters
    assert is_isogram('aa') == False  # 'a' is repeated
    assert is_isogram('') == False  # Empty string is not an isogram
    assert is_isogram('12345') == True  # Numeric characters, all unique
    assert is_isogram('12334') == False  # '3' is repeated