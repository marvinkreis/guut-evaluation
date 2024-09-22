from string_utils._regex import PRETTIFY_RE

def test_prettify_re():
    # Valid string that should produce matches for duplicates
    test_string_valid = "This   is   a   test   string."  # Multiple spaces should trigger a match

    # Invalid string that should not produce matches for the duplicates
    test_string_invalid = "This is a test string."  # No duplicates in spaces

    # Using precompiled regex from PRETTIFY_RE to check behavior
    result_valid = PRETTIFY_RE['DUPLICATES'].search(test_string_valid)
    result_invalid = PRETTIFY_RE['DUPLICATES'].search(test_string_invalid)

    # Correct code should find a match in the valid string and none in the invalid
    assert result_valid is not None  # Should find matches on the test string with multiple spaces
    assert result_invalid is None     # Should not find matches on the regular string