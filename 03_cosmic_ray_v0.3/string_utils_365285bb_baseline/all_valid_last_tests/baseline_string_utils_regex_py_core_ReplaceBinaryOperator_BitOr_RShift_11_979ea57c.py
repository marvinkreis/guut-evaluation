from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test case 1: Detect multiple spaces in a string
    test_string = "This  is  a  test  string with  multiple spaces."
    matches_duplicates = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    assert len(matches_duplicates) >= 2, f"Expected at least 2 matches for duplicates, found {len(matches_duplicates)}."
    
    # Test case 2: Check quoted string with excessive spaces
    quoted_string = '"This   is a  quoted  sentence."'
    quoted_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(quoted_string)
    assert len(quoted_matches) == 1, "Expected to find matched quoted text but found none."

    # Test case 3: Check bracketed string with spaces
    bracket_string = '(This  is    inside )'
    bracket_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(bracket_string)
    assert len(bracket_matches) == 1, "Expected to find a match within brackets but found none."

    # Test case 4: Spaces around punctuation—clear wrong capture check
    punctuation_check = 'Is there an  issue  here?'
    right_space_matches = PRETTIFY_RE['RIGHT_SPACE'].findall(punctuation_check)
    assert len(right_space_matches) == 0, "Expected no capture of spaces around punctuation."
    
    # Test case 5: Check improper formatting on text
    improper_format_string = 'Text with    unneeded   extra   spaces.'
    improper_format_matches = PRETTIFY_RE['DUPLICATES'].findall(improper_format_string)
    assert len(improper_format_matches) >= 1, "Expected to find captures of excessive spaces, but none found."

    # Test case 6: Check handling of strange formats like new lines (should differ under mutation)
    strange_format_string = 'This   is   line one.\nThis is line two!   '
    strange_format_matches = PRETTIFY_RE['DUPLICATES'].findall(strange_format_string)
    assert len(strange_format_matches) >= 1, "Expected to see captures due to strange formatting in multiple lines."

# Ensure to run if being executed directly for debugging
if __name__ == '__main__':
    test__prettify_re()