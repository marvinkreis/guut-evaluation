from string_utils.manipulation import prettify

def test__prettify():
    """This test will highlight the flaw in the mutant's handling of spaces and punctuation."""
    # Complex input with various spacing/formatting irregularities
    input_string = 'Héllo!!!   This is a test...   What’s happening?   '
    expected_output = 'Héllo!!! This is a test... What’s happening?'  # Correctly prettified output

    output = prettify(input_string)

    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"