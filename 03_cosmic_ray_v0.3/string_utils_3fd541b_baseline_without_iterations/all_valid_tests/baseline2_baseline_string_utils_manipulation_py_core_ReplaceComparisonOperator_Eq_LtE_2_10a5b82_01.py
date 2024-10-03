from string_utils.manipulation import roman_encode

def test__roman_encode_five():
    # This test checks the encoding of the number 5 to Roman numeral
    result = roman_encode(5)
    assert result == 'V', f"Expected 'V' but got {result}"

    # Additional test cases
    result_two = roman_encode(4)
    assert result_two == 'IV', f"Expected 'IV' but got {result_two}"

    result_three = roman_encode(6)
    assert result_three == 'VI', f"Expected 'VI' but got {result_three}"