from string_utils._regex import PRETTIFY_RE

def test_uppercase_after_sign():
    # Input string that should match the original regex
    input_string = "Hello world! This is a test." 
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(input_string) is not None  # Should find 'T' in 'This'

    # A complex case that includes other spacing and flags that should show mutant difference
    input_string_mutant = "Hello world!   This is a test."  # Multiple spaces after '!'
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(input_string_mutant) is None  # Should not find 'T' with mutant regex

# This test should pass for the original code and fail for the mutant due to regex differences.