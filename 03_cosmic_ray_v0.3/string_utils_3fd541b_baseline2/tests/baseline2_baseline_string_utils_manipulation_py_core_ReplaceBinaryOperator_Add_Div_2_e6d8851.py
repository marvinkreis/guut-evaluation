from string_utils.manipulation import roman_encode

def test_roman_encode():
    # This test will check the functionality of roman_encode with known inputs
    # It should pass for the original implementation and fail for the mutant.

    # Test case for normal conversion
    assert roman_encode(9) == 'IX', "Expected 'IX' for input 9"
    
    # Test case for boundary value
    assert roman_encode(4) == 'IV', "Expected 'IV' for input 4"
    
    # Test case for larger number
    assert roman_encode(3999) == 'MMMCMXCIX', "Expected 'MMMCMXCIX' for input 3999"
    
    # Test case for zero (should raise an exception)
    try:
        roman_encode(0)
        assert False, "Expected ValueError for input 0"
    except ValueError:
        pass  # This is the expected outcome
    
    # Test case for out of bounds input (should raise an exception)
    try:
        roman_encode(4000)
        assert False, "Expected ValueError for input 4000"
    except ValueError:
        pass  # This is the expected outcome