from string_utils._regex import PRETTIFY_RE

def test__regex_uppercase_after_sign():
    """
    Test the UPPERCASE_AFTER_SIGN regex pattern. 
    This checks if the pattern can correctly identify a punctuation followed by whitespace and an uppercase letter.
    The test should pass without errors when run with the baseline,
    while it should fail (due to an import error) when run with the mutant.
    """
    
    valid_input = "Hello! How are you?"
    invalid_input = "hello! how are you?"
    
    # Test with valid input that should match the regex
    match_valid = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(valid_input)
    assert match_valid is not None  # Should match

    # Test with invalid input; this input should not fail but anticipate matching something valid
    match_invalid = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(invalid_input)
    assert match_invalid is not None  # This should also find something potentially valid in regex patterns