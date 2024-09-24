from string_utils._regex import PRETTIFY_RE

def test_prettify_re():
    
    # Test input where duplicates should definitely be recognized
    test_string = "This is a test...     and here is a test...."
    
    # Find matches for duplicates using the regex defined
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # Expect the correct implementation to find duplicates
    assert len(matches) > 0, "Expected to find duplicate matches in the correct code."

    # Test with an input string that includes multiple lines with duplicates
    multiline_string = """
    This line has some excessive spaces... and poor punctuation.....
    Another line.... here is another test... and another test...!
    """
    
    # Find matches in multiline context
    multiline_matches = PRETTIFY_RE['DUPLICATES'].findall(multiline_string)
    assert len(multiline_matches) > 0, "Expected to find duplicates in the multiline input."

    # Check an input that has no duplicates
    clean_string = "This string has no duplicates in it."
    no_matches = PRETTIFY_RE['DUPLICATES'].findall(clean_string)

    # Ensure that no matches are found where there are no duplicates
    assert len(no_matches) == 0, "Expected to find no duplicates in a clean string."