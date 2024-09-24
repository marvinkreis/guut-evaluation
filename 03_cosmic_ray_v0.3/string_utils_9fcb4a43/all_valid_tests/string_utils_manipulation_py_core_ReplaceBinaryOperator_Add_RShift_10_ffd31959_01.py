from string_utils.manipulation import prettify

def test__prettify():
    """
    This test uses an input that improperly utilizes spaces and punctuation,
    which should expose discrepancies in how the mutant formats the output 
    due to its flawed handling of string formatting.
    """
    # Input string with irregular spaces and punctuations
    input_string = '   What   is   this   ?   This   is   a   test   !   '
    
    # The expected output should not have extra spaces around punctuation
    expected_output = 'What is this? This is a test!'
    
    # Get the output from the prettify function
    output = prettify(input_string)
    
    # Assert: Check if the output matches the expected formatting
    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"