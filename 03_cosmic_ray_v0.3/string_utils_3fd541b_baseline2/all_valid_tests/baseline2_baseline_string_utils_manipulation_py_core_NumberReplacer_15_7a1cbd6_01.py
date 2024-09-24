from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test the correct case
    assert roman_encode(1) == 'I'  # 1 should be correctly encoded to 'I'

    # Test invalid input, should raise ValueError (zero is invalid input)
    try:
        roman_encode(0)
        assert False, "Expected ValueError not raised for input 0"
    except ValueError:
        pass  # This is expected behavior
    
    # Providing a valid input of 2 to ensure normal functioning
    assert roman_encode(2) == 'II'  # 2 should be encoded to 'II'