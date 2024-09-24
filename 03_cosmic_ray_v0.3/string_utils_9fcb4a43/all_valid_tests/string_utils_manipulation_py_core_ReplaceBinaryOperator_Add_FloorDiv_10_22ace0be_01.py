from string_utils.manipulation import prettify

def test__prettify():
    """The mutant change will cause the prettify function to incorrectly handle spaces around punctuation."""
    input_string = '   Wow!!!   Amazing....      Look at that!!!    '
    expected_output = 'Wow!!! Amazing.... Look at that!!!'  # Correctly prettified output
    output = prettify(input_string)
    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"