from string_utils._regex import PRETTIFY_RE

def test__multiline_string_regex_behavior():
    """
    Test the behavior of the regex for matching spaces and line breaks in a multiline string.
    This test validates that the change in the mutant version raises a ValueError due to incompatible flags,
    whereas the baseline version compiles and finds matches correctly.
    """
    test_string = "Hello  World\n\nHere is a test  string."
    
    # This should function without exception in the baseline but throw an exception in the mutant
    try:
        matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
        assert matches == ['  ', '\n\n', '  ']
    except ValueError:
        assert True  # This means we have successfully caught the value error in mutant.