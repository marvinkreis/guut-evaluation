from string_utils.manipulation import roman_encode

def test__roman_encode_zero():
    """Inputting zero to roman_encode should raise a ValueError in the correct implementation but not in the mutant."""
    try:
        roman_encode(0)
        assert False, "Expected ValueError for input 0, but no exception was raised."
    except ValueError:
        assert True  # ValueError is expected

# Running the test
test__roman_encode_zero()