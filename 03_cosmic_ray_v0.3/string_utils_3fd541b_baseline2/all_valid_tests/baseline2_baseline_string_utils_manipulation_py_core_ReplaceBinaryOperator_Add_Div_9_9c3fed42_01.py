from string_utils.manipulation import prettify

def test__prettify__detect_mutant():
    # The test string that contains various spacing issues
    input_string = "Hello   World!   This is a test.   "
    
    # The expected output with correct implementation
    expected_output = "Hello World! This is a test."

    # Assert that the prettify function output matches the expected result
    assert prettify(input_string) == expected_output, "The output does not match the expected formatted string."

# Execute the test to see if it passes or fails
test__prettify__detect_mutant()