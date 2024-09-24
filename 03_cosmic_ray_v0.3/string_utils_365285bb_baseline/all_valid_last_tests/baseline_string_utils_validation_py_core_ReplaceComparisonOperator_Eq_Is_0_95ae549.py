from string_utils.validation import is_isogram

def test_is_isogram():
    # Test valid isogram inputs
    assert is_isogram('dermatoglyphics') == True          # should return True
    assert is_isogram('abcdefg') == True                   # should return True
    assert is_isogram('123456') == True                    # should return True
    assert is_isogram('!@#$%^&*()') == True                 # should return True

    # Test invalid isogram inputs
    assert is_isogram('hello') == False                    # should return False
    assert is_isogram('aA') == True                         # should return True (case sensitivity)
    assert is_isogram('abcdeff') == False                  # should return False (f is repeated)
    assert is_isogram('123321') == False                    # should return False (1 and 2 are repeated)

    # A long isogram string
    assert is_isogram('abcdefghij') == True                 # should return True (unique characters)
    # A long string with a known repeat
    assert is_isogram('abcdefghijj') == False               # should return False (j is repeated)

# Run the test
test_is_isogram()