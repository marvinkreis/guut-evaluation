from string_utils._regex import PRETTIFY_RE

def test__upper_case_after_sign():
    # Input string that should match the UPPERCASE_AFTER_SIGN regex correctly
    test_string_correct = "Hello! How are you?"
    
    # Using the correct regex should find a match here for "H"
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string_correct) is not None, "Test should pass with correct code"
    
    # Input string that should not match the UPPERCASE_AFTER_SIGN regex
    test_string_mutant = "Hello!how are you?"  # Missing space after '!'
    
    # This should fail if the mutant is applied, as it will incorrectly return a match for the first character of 'how'
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string_mutant) is None, "Test should fail with the mutant"