from string_utils.manipulation import roman_encode

def test_roman_encode_kill_mutant():
    """
    Test the roman_encode function with an input that will reveal the mutant's behavior.
    The input 44 requires encoding that involves the tens place, specifically testing the mutated mapping.
    The baseline should return 'XLIV', while the mutant raises a KeyError.
    """
    output = roman_encode(44)
    assert output == 'XLIV', f"Expected 'XLIV', got {output}"