from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's change may allow for roman_encode(3999) to return a value instead of raising an error."""
    try:
        output = roman_encode(3999)  # This should throw an error in correct implementation
        assert isinstance(output, str), "Expected output to be a string for the mutant implementation"
    except ValueError:
        assert False, "Expected ValueError for input 3999"
    
    # Confirm valid Roman numeral for mutant
    assert output == "MMMCMXCIX", "Output did not match expected Roman numeral for mutant"

# Uncomment to run the test
# test__roman_encode()