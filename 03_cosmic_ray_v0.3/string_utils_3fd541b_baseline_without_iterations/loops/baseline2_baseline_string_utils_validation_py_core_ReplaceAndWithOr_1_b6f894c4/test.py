from string_utils.validation import is_integer

def test__is_integer():
    # This input is expected to return True because '42' is an integer
    assert is_integer('42') == True
    # This input is expected to return False because '42.0' is not an integer
    assert is_integer('42.0') == False
    # Additional input that should return False - it's a decimal number
    assert is_integer('3.14159') == False
    # Additional input that should return False - is a string that contains letters
    assert is_integer('forty two') == False
    # Additional input that should return True - a negative integer
    assert is_integer('-10') == True
    # Additional input that should return True - zero is an integer
    assert is_integer('0') == True