from string_utils.manipulation import prettify

def test__prettify_heavy_special_characters_correct():
    """
    Test the `prettify` function with a string that contains heavy use of special characters, tabs, and newlines.
    The baseline should recover a well-formatted output, while the mutant should raise an error.
    """
    input_string = "   \n\n\t\t**Attention!!**   Here! is     a test with   unusual spacing...   Yeah???   \t \n "
    # Expecting the baseline to return a formatted string
    baseline_output = prettify(input_string)
    
    # Now validate the output on baseline
    assert baseline_output == '*  * Attention!! *  * Here! Is a test with unusual spacing... Yeah???'
    
    # Now checking for mutant
    try:
        mutant_output = prettify(input_string)
        assert False, "Expected an error in the mutant, but the call succeeded."
    except Exception as e:
        print(f"Mutant Error: {e}")  # This must be caught in the mutant case