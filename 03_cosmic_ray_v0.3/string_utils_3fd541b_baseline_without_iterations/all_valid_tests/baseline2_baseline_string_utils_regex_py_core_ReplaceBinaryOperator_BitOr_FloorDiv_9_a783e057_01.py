from string_utils._regex import PRETTIFY_RE

def test__UPPERCASE_AFTER_SIGN_mutant_detection():
    # Test strings that should detect the regex behavior
    test_string_valid = "Hello! How are you?"
    test_string_not_valid = "This is a test.hello."

    # Validate that the PRETTIFY_RE 'UPPERCASE_AFTER_SIGN' correctly identifies when a character must be followed by uppercase letters
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string_valid) is not None, "The valid test string should match the regex."
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string_not_valid) is None, "The invalid test string should not match the regex."