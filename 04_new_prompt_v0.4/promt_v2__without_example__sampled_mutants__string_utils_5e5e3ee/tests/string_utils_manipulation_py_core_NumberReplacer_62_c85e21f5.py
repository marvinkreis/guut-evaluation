from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_whitespace_kills_mutant():
    """
    Test the prettify function on a string with irregular whitespaces.
    This test is designed to kill the mutant by producing an IndexError in the mutant code
    while the baseline will handle it correctly and return a formatted string. 
    The input contains excessive leading and trailing whitespace which should be normalized.
    """
    input_string = '   Hello    World!   '  # This input contains irregular spaces.
    
    expected_output_baseline = 'Hello World!'  # The expected output from the baseline.

    output = prettify(input_string)  # This should execute without errors on baseline.
    print(f"output: {output}")
     
    assert output == expected_output_baseline  # This assertion will pass with the baseline.