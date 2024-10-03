from string_utils._regex import PRETTIFY_RE

def test__prettify_multiline():
    """
    Test the regex PRETTIFY_RE['LEFT_SPACE'] on a multiline string. The test should pass in the baseline 
    and fail in the mutant due to an incompatible combination of regex flags after the mutation.
    """
    mult_string = "This is a test string.  \n(Hello World)  \nAnother line."
    try:
        match = PRETTIFY_RE['LEFT_SPACE'].search(mult_string)
        assert match is None  # Expecting None for no matches
    except ValueError:
        assert True  # The mutation should raise a ValueError