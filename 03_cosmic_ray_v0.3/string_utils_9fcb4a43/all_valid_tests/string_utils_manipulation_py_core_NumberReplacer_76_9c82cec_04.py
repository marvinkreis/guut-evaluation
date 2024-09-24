from string_utils.manipulation import prettify

def test__prettify():
    """Test for the behavior of prettify with leading, trailing, and interspersed spaces."""
    input_string = "     Hello,     World!     "  # Leading and trailing spaces, excessive spaces within
    output = prettify(input_string)
    expected_output = "Hello, World!"  # Expected correct formatting without extra spaces
    
    # Assert to check for expected behavior
    assert output == expected_output, "prettify should format the string correctly."