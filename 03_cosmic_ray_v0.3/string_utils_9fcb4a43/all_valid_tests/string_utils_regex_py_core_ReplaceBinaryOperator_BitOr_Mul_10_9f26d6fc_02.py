from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """The mutant changes functionality and should not match against repeated characters."""
    # Test input that contains duplicate characters to see if the regex can capture these patterns.
    test_input = 'This is an example  (with  multiple spaces   and  periods.....'
    
    # Using the DUPLICATES regex rule from the PRETTIFY_RE
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_input)
    
    # We should expect at least one match for the repeated periods.
    assert len(matches) > 0, "PRETTIFY_RE DUPLICATES must find matches in the input string."