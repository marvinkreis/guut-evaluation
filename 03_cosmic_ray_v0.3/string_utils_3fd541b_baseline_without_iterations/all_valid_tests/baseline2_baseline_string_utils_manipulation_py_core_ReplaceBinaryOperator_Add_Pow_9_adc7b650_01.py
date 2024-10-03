from string_utils.manipulation import prettify

def test__prettify():
    # Test if prettify correctly formats spaces around text
    input_string = " Hello World! This is a test. "
    expected_output = "Hello World! This is a test."
    actual_output = prettify(input_string)
    
    # The test should fail with mutant because it incorrectly modifies spaces
    assert actual_output == expected_output, f"Expected: {expected_output}, but got: {actual_output}"