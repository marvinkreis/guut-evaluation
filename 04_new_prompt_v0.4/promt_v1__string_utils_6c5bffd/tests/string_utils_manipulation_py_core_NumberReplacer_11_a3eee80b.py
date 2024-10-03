from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function for specific cases that validate Roman numeral representations.
    The test checks if encoding the integer 400 produces 'CD' on the baseline,
    while it should raise a KeyError on the mutant due to the change in the mapping for hundreds.
    """
    # This should pass on the baseline
    assert roman_encode(400) == 'CD'
    # This should also pass on the baseline
    assert roman_encode(100) == 'C'
    # This should also pass on the baseline
    assert roman_encode(500) == 'D'