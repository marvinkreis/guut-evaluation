from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 3999. The baseline should return
    the Roman numeral 'MMMCMXCIX', while the mutant will raise a ValueError
    because it incorrectly limits the input to 3998.
    """
    try:
        output = roman_encode(3999)
        assert output == 'MMMCMXCIX', f"Expected 'MMMCMXCIX', got {output}"
    except ValueError as e:
        # This block should not be executed on the baseline
        assert False, f"Should not raise ValueError for 3999: {e}"