from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    # Test string containing valid patterns
    test_string = 'Hello, this is a test string with quotes "this is quoted text" and (these are brackets).'
    
    # The following should match the quoted text and text in brackets
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)
    
    # The expected matches should contain the quoted text and the text in brackets
    expected_matches = ['this is quoted text', 'these are brackets']

    assert matches == expected_matches, f"Expected {expected_matches}, but got {matches}"

    # Now testing with an invalid case where the mutant will fail
    mutant_test_string = 'Hello, this is a test string with no spaces around symbols !?'
    mutant_matches = PRETTIFY_RE['SPACES_AROUND'].findall(mutant_test_string)
    
    # In the original code, this should only return symbols that are misplaced
    expected_mutant_matches = ['!', '?']  # Assuming that the original regex would identify this
    
    # Asserting the output of the mutant based on the faulty regex
    assert mutant_matches != expected_mutant_matches, "The mutant should mismatch this pattern."
