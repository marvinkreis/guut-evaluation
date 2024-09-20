from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant fails to correctly encode 500 or any number that requires 'D' due to a KeyError."""
    try:
        assert roman_encode(500) == 'D', "Expected 'D' for input 500"
        assert roman_encode(501) == 'DI', "Expected 'DI' for input 501"
    except KeyError:
        assert False, "Mutant failed by raising KeyError on valid input"

test__roman_encode()