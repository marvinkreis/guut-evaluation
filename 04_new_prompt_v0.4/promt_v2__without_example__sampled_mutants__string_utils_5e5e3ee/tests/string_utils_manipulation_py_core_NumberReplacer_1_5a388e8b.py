from string_utils.manipulation import roman_encode

def test__roman_encode_units():
    """
    Test the roman_encode function for unit values 1, 2, and 3.
    The expected results are 'I', 'II', and 'III' respectively based on proper Roman numeral encoding.
    The mutant should fail to encode these correctly due to the change made in mappings.
    """
    output_1 = roman_encode(1)
    output_2 = roman_encode(2)
    output_3 = roman_encode(3)

    # Assertions to confirm expected behavior
    assert output_1 == 'I', f"Expected 'I' but got {output_1}"
    assert output_2 == 'II', f"Expected 'II' but got {output_2}"
    assert output_3 == 'III', f"Expected 'III' but got {output_3}"

test__roman_encode_units()