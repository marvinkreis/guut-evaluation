from string_utils.manipulation import prettify

def test__prettify_ensure_spaces_around():
    # The input string is intentionally formatted to test how the __ensure_spaces_around function works.
    input_string = "Hello    World!  This   is a test string."
    expected_output = "Hello World! This is a test string."  # Expected output with single spaces

    # Use the prettify function to process the input
    actual_output = prettify(input_string)
    
    # Assert to check the output is as expected
    assert actual_output == expected_output, f"Expected: '{expected_output}', but got: '{actual_output}'"