from string_utils.validation import is_isogram

def test__is_isogram():
    # Test cases for is_isogram
    assert is_isogram('dermatoglyphics') == True  # It is an isogram
    assert is_isogram('hello') == False  # It is not an isogram
    assert is_isogram('abcdefg') == True  # It is an isogram
    assert is_isogram('123456') == True  # It is an isogram
    assert is_isogram('!@#$%^&*()') == True  # Special characters, should return True
    assert is_isogram('') == False  # An empty string should not be considered an isogram
    assert is_isogram('abcdeff') == False  # It is not an isogram due to repeated 'f'