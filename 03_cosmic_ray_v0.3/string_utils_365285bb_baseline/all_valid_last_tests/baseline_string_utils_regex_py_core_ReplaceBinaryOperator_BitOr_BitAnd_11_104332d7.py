from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Test case: A properly formatted quoted string which should be matched.
    test_string = '"This is a valid quoted text"'
    
    # The original regex `PRETTIFY_RE['SPACES_INSIDE']` should match properly.
    match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    
    # The mutant version should fail to match since it uses & instead of |,
    # thus failing to allow the regex to match quoted text as intended.
    assert match is not None, "Expected match for the original regex"
    
    # You can also test a string that doesn't conform to the matching,
    # which should still work for both implementations:
    invalid_test_string = "No quotes here!"
    invalid_match = PRETTIFY_RE['SPACES_INSIDE'].search(invalid_test_string)
    assert invalid_match is None, "Expected no match for invalid string"