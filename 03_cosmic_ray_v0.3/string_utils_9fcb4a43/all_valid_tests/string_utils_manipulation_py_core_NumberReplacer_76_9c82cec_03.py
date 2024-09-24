from string_utils.manipulation import prettify

def test__prettify():
    """Test for the behavior of prettify with multiple spaces around words."""
    input_string = "Hello  world!  This is   a test."  # Test input with irregular spacing
    output = prettify(input_string)
    expected_output = "Hello world! This is a test."  # Expected correct formatting
    
    # Assert to check for expected behavior
    assert output == expected_output, "prettify should format the string correctly."