from string_utils.manipulation import roman_encode

def test__roman_encode_for_four():
    """
    Test the roman encoding of the integer 4. The original implementation should correctly return "IV", while the mutant 
    has a changed mapping causing it to potentially throw an exception (KeyError). This differentiates the behavior
    between the baseline and the mutant.
    """
    output = roman_encode(4)
    assert output == "IV"  # expected output for roman encoding of 4