from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE_duplicates_matching():
    """Ensure that the correct implementation of PRETTIFY_RE can match duplicates."""
    test_input = "Hello, how are you??\n\n\nI am fine.  Too fine!"
    
    # Using the 'DUPLICATES' regex pattern from PRETTIFY_RE
    match = PRETTIFY_RE['DUPLICATES'].search(test_input)
    
    if match:
        print(f"Match found: {match.group()}")
    else:
        print("No match found!")
    
    assert match is not None, "Correct PRETTIFY_RE should match the input string with duplicates."

# Execute the test
test__PRETTIFY_RE_duplicates_matching()