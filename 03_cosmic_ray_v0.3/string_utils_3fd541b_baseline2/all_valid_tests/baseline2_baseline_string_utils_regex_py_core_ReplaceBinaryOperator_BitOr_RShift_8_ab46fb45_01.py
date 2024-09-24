from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # This input should not raise an error when compiled
    test_string = "This is a test sentence, with some   extra spaces."
    
    # The regex should match extra spaces that should not be repeated
    result = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # We expect to find one match for the extra spaces
    assert len(result) == 1

    # Check for specific match (for dupliacted spaces)
    assert result[0] == "   "  # We expect the duplicated spaces as the match

# The test__prettify_re function should pass with the original code
# and fail with the mutant due to an error in regex compilation.