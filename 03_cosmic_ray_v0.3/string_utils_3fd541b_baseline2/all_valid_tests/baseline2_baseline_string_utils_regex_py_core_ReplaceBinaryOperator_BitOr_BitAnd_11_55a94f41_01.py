from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test strings
    test_string = '"Hello World!"'
    
    # The regex should match quoted text, so we use the pattern for 'SPACES_INSIDE'
    match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    
    # We expect a match with the quoted text
    assert match is not None, "Expected a match for quoted text"
    
    # Set up a test case that would fail if the mutant is applied
    mutant_string = 'Hello World!'  # No quotes, we expect no match
    mutant_match = PRETTIFY_RE['SPACES_INSIDE'].search(mutant_string)
    
    # We expect no match with the mutant string
    assert mutant_match is None, "Expected no match for non-quoted text"