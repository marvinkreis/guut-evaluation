from string_utils.manipulation import prettify

def test__prettify():
    # Input should have spaces before the period to test the prettify formatting
    input_string = 'Here is a correctly formatted sentence but with too many spaces before the end      .'
    expected_output_with_correct_code = 'Here is a correctly formatted sentence but with too many spaces before the end.'

    # Execute the prettify function and check if the output is as expected.
    output = prettify(input_string)

    # Check the output against the expectation
    assert output == expected_output_with_correct_code, f'Expected "{expected_output_with_correct_code}", but got "{output}"'