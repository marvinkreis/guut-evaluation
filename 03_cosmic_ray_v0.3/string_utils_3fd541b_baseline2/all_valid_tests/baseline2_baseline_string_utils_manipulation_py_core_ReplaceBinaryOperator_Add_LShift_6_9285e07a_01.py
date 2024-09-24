from string_utils.manipulation import prettify

def test_prettify():
    # Test case that checks the prettify function when it encounters an uppercase letter after a sign.
    input_string = 'hello! this is a test.'
    expected_output = 'Hello! This is a test.'  # The first character after '!' should be uppercase.

    # Call the prettify function
    output = prettify(input_string)

    # Assert that the output is as expected
    assert output == expected_output, f'Expected "{expected_output}", but got "{output}".'

    # Test to trigger the mutant behavior
    malformed_input_string = 'foo! bar. this should work.'  # This tests another case.
    should_work_output = 'Foo! Bar. This should work.'  # Expected correct output

    # Call the prettify function to ensure the method works correctly as well
    malformed_output = prettify(malformed_input_string)

    # Assert that the output is as expected
    assert malformed_output == should_work_output, f'Expected "{should_work_output}", but got "{malformed_output}".'