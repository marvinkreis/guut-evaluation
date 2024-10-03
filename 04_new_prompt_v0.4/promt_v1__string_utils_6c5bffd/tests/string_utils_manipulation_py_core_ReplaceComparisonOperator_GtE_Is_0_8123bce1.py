from string_utils.manipulation import roman_decode

def test__roman_decode_final():
    """
    This test checks that the Roman numeral decodings yield the correct integer values.
    The input 'MCMXCIV' should decode to 1994 and 'MMXXI' should decode to 2021.
    The mutant's outputs will be incorrect due to the change in comparison logic.
    """
    assert roman_decode('MCMXCIV') == 1994, "Expected 1994 for 'MCMXCIV'"
    assert roman_decode('MMXXI') == 2021, "Expected 2021 for 'MMXXI'"