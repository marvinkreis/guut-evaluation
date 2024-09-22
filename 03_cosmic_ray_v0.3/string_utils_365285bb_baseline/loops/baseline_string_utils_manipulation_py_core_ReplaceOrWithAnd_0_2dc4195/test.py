from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test a valid input value (should work with the original function)
    assert roman_encode(1) == 'I'
    assert roman_encode(3999) == 'MMMCMXCIX'
    
    # Test an invalid input value (greater than 3999)
    try:
        roman_encode(4000)
        assert False, "Expected ValueError for input greater than 3999"
    except ValueError:
        pass  # This is expected for the original code

    # Test an invalid input value (less than 1)
    try:
        roman_encode(0)
        assert False, "Expected ValueError for input less than 1"
    except ValueError:
        pass  # This is expected for the original code
    
    # Test an invalid input value (negative)
    try:
        roman_encode(-1)
        assert False, "Expected ValueError for negative input"
    except ValueError:
        pass  # This is expected for the original code