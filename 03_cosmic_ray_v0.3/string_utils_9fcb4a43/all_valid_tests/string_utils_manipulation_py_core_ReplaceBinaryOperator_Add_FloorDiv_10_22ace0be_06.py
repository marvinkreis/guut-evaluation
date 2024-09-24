from string_utils.manipulation import prettify

def test__prettify():
    """Test designed to expose mutant's logical flaws in space handling."""
    
    # Input string deliberately structured to break expected formatting
    input_string = ' Hello   Wow!!!   What?   Are you   done?  '
    expected_output = 'Hello Wow!!! What? Are you done?'  # Expected to prettify correctly
    
    output = prettify(input_string)

    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"