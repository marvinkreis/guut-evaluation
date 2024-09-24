from string_utils.manipulation import prettify

def test__prettify():
    """Tests the prettify function to reveal mutant differences."""
    
    # Edge case with intricate spacing and punctuation
    input_string = '   Example:    Hello!!!   What is this?    Are you    serious...    '
    expected_output = 'Example: Hello!!! What is this? Are you serious...'  # Correct formatting
    
    output = prettify(input_string)

    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"