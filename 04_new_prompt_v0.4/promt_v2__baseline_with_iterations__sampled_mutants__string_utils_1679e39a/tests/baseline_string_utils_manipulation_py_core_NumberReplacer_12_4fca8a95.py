from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with the input 1000. The expected Roman numeral for 1000 is 'M'.
    The baseline will return 'M' correctly, but the mutant's change alters the thousands encoding logic,
    which may lead to incorrect output for higher values.
    This input is specifically chosen to test the mutant that incorrectly sets the mapping for thousands.
    """
    output = roman_encode(1000)
    assert output == 'M'  # The baseline should return 'M', but the mutant would not return the expected value.