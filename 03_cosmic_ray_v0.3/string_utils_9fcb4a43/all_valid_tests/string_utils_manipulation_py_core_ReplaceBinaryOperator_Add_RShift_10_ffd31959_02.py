from string_utils.manipulation import prettify

def test__prettify():
    """
    This test checks the pretty formatting of a complex string with strange punctuation and spaces.
    The mutant's changes should lead to distinct output by mishandling punctuation placement.
    """
    # Input string with odd spaces around punctuation
    input_string = '   Hello   ?   How  are   you   !   This   is   a     test?   '
    
    # The expected output should have correctly placed punctuation with proper spacing.
    expected_output = 'Hello? How are you! This is a test?'
    
    # Get the output from the prettify function
    output = prettify(input_string)
    
    # Assert: Check if the output matches the expected formatting
    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"