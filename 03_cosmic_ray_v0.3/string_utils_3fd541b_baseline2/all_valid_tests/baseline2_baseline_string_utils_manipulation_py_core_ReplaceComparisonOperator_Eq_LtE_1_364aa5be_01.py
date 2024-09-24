from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Testing the behavior for the value 4
    # The correct output for roman_encode(4) should be "IV"
    # Under the mutant, roman_encode(4) will incorrectly return "IIII"
    assert roman_encode(4) == "IV", "Expected 'IV' for the input 4"
    
    # Additionally testing the bounds
    # The correct output for roman_encode(1) should be "I"
    # The correct output for roman_encode(3999) should be "MMMCMXCIX"
    assert roman_encode(1) == "I", "Expected 'I' for the input 1"
    assert roman_encode(3999) == "MMMCMXCIX", "Expected 'MMMCMXCIX' for the input 3999"

    # Let's also include an ascending sequence check,
    # for numbers 1 to 10, the expected roman numerals are:
    expected_romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for i in range(1, 11):
        assert roman_encode(i) == expected_romans[i - 1], f"Expected '{expected_romans[i - 1]}' for the input {i}"