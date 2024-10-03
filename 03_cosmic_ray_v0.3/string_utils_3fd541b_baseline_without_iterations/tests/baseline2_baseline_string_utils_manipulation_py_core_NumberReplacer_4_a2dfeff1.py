from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test the correct encoding of a value
    assert roman_encode(10) == 'X', "Encoding of 10 should be 'X'"
    assert roman_encode(37) == 'XXXVII', "Encoding of 37 should be 'XXXVII'"
    assert roman_encode(58) == 'LVIII', "Encoding of 58 should be 'LVIII'"
    assert roman_encode(1994) == 'MCMXCIV', "Encoding of 1994 should be 'MCMXCIV'"
    
    # This specifically tests the affected range due to the mutant change.
    assert roman_encode(20) == 'XX', "Encoding of 20 should be 'XX'"  # This will fail with the mutant

# The assertion for the value 20 is key because the mutant's change affects how tens are treated,
# causing the output to be incorrect.