from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for normal Roman numeral encoding
    assert roman_encode(1) == 'I', "Failed on input 1"
    assert roman_encode(2) == 'II', "Failed on input 2"
    assert roman_encode(3) == 'III', "Failed on input 3"
    assert roman_encode(4) == 'IV', "Failed on input 4"
    assert roman_encode(5) == 'V', "Failed on input 5"
    assert roman_encode(10) == 'X', "Failed on input 10"
    assert roman_encode(37) == 'XXXVII', "Failed on input 37"
    assert roman_encode(2020) == 'MMXX', "Failed on input 2020"
    
    # Test case which specifically targets the mutant
    assert roman_encode(3999) == 'MMMCMXCIX', "Failed on input 3999"
    
    print("All tests passed!")

# Uncomment the line below to run the test directly
# test_roman_encode()