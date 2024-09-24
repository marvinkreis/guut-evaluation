from string_utils.manipulation import prettify

def test_prettify():
    # Input that includes repeated phrases where double replacements will show issues
    input_string = "This is a test. This is a test. This is another test."

    # The expected output should return without altering duplicate sentences
    expected_output = "This is a test. This is a test. This is another test."

    # Execute the prettify function
    actual_output = prettify(input_string)

    # Assert that the actual output matches the expected output and check for discrepancies
    assert actual_output == expected_output, f'Expected: "{expected_output}", but got: "{actual_output}"'

# Uncomment below to run the test
# test_prettify()