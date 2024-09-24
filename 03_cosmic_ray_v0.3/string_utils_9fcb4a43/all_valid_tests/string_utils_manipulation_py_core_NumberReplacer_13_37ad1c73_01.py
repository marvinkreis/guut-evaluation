from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Changing the mapping in RomanNumbers from {1: 'M'} to {0: 'M'} will result in a KeyError for input >= 1000."""
    try:
        result = roman_encode(1000)
        assert result == 'M', f"Expected 'M' but got {result} instead."
    except KeyError as e:
        # This indicates the mutant has taken effect.
        print(f"KeyError raised as expected: {e}")
        assert False, "Expected a valid output but received a KeyError instead."