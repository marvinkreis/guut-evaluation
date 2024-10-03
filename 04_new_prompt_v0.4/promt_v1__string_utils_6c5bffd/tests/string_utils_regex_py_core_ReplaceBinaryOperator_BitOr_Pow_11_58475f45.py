from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_kills_mutant():
    """
    Test the SPACES_INSIDE regular expression to verify that it correctly matches 
    quoted text. The mutant version causes an OverflowError due to incorrect operator
    usage (using ** instead of |). This test checks for the correct behavior of 
    the Baseline and ensures it produces matches while the Mutant raises an error.
    """
    test_string = '"hello world"'
    try:
        matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)
        assert matches == ['hello world'], "Expected match not found"
    except Exception as e:
        print(f"An exception occurred: {e}")
        assert False  # Ensure the test fails if an exception is raised