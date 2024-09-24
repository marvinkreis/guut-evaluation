from string_utils.manipulation import prettify

def test__prettify():
    """This test aims to expose the mutant's handling of spaces and punctuation."""
    # Input with various irregularities
    input_string = '   Hello!!!   How are you?     I hope   you   are   doing well!!!   '
    expected_output = 'Hello!!! How are you? I hope you are doing well!!!'  # The expected prettified output

    output = prettify(input_string)

    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"