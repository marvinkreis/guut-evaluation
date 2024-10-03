from string_utils._regex import PRETTIFY_RE

def test__prettify_duplicates():
    # Original regex expects to find duplicated spaces.
    test_string = "This    is a test string with  multiple spaces."
    
    # The expected behavior is to match the excess spaces.
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # We expect matches in the case of duplicated spaces
    assert len(matches) > 0, "Expected to find duplicated spaces, but found none."

    # Check if the matches contain the correct duplicate pattern
    assert any("  " in match for match in matches), "Expected at least one match for duplicated spaces."

# Call the test function to ensure it's functioning correctly
test__prettify_duplicates()