from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test for the number 1, which should return 'I'
    output = roman_encode(1)
    expected_output = 'I'
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"

    # Test for the number 2, which should return 'II'
    output = roman_encode(2)
    expected_output = 'II'
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"

    # Test for the number 3, which should return 'III'
    output = roman_encode(3)
    expected_output = 'III'
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"

    # Test for number 4, which should return 'IV'
    output = roman_encode(4)
    expected_output = 'IV'
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"

    # Test for number 5, which should return 'V'
    output = roman_encode(5)
    expected_output = 'V'
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"