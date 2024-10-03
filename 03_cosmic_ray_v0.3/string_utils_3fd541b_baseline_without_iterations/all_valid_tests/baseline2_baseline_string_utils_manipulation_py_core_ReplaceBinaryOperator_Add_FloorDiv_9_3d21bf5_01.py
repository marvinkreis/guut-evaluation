from string_utils.manipulation import prettify

def test__prettify():
    # Test input where we expect spaces to be around the string
    input_string = " test string "
    expected_output = "Test string"
    
    # Assert that the function works correctly with the original code
    assert prettify(input_string) == expected_output
    
    # Mutant will fail this test due to incorrect string concatenation