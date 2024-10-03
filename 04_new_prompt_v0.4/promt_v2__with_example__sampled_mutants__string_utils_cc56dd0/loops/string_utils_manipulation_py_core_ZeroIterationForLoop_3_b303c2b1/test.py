from string_utils.manipulation import prettify

def test_prettify_mutant_killing():
    """
    Test the prettify function with a string containing a URL and an email.
    The baseline will return the original string formatted correctly, while the mutant will distort it.
    """
    original_string = 'Visit us at https://example.com or contact us at info@example.com.'
    expected_output = 'Visit us at https://example.com or contact us at info@example.com.'

    output = prettify(original_string)
    
    # The mutant will distort the output, so we assert that it is equal to the expected output.
    assert output == expected_output, f"Expected: {expected_output}, but got: {output}"