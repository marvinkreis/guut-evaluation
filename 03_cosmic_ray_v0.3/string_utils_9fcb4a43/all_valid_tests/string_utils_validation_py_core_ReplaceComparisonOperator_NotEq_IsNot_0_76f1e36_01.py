from string_utils.validation import is_full_string

def test__is_full_string_variants():
    """Checking multiple cases for distinguishing behavior."""
    # Each case should return false for empty or whitespace
    assert not is_full_string(" "), "Single space should not be full."
    assert not is_full_string(""), "Empty string should not be full."
    assert not is_full_string(" \t\n "), "Whitespace string should not be full."
    
    # Valid cases that should return true
    assert is_full_string("abc"), "Regular string should be full."
    assert is_full_string("string with space"), "Non-empty string with space should be full."
    assert is_full_string(" a "), "Valid string with leading and trailing space should be full."
    assert is_full_string("0"), "String '0' should be considered full."