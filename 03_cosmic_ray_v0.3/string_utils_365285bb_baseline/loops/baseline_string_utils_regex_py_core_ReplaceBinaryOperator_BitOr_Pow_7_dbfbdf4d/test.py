from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # This text has multiple spaces and should be simplified
    test_text = "This is   a    test    string with multiple   spaces."
    
    # The regex should match repeated spaces, which should be considered as duplicates
    # 'DUPLICATES' in PRETTIFY_RE should find multiple spaces and potentially match
    # the string.

    # Using findall to find all duplicates
    duplicates_found = PRETTIFY_RE['DUPLICATES'].findall(test_text)
    
    # If the code is correct, it should find that there are duplicates.
    # The expected output should have multiple space occurrences accounted,
    # which will return a list length greater than 0.
    assert len(duplicates_found) > 0, "The regex should detect duplicate spaces."