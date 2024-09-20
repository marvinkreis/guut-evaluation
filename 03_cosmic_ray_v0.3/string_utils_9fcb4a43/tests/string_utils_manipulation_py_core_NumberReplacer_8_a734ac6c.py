from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant has faulty mappings for encoding roman numbers and should fail tests for numbers with hundreds."""
    
    # Test cases based on the mappings
    assert roman_encode(200) == 'CC', "roman_encode(200) should return 'CC'"
    assert roman_encode(300) == 'CCC', "roman_encode(300) should return 'CCC'"

# Call the test function
test__roman_encode()