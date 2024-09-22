from string_utils._regex import PRETTIFY_RE

def test__upper_case_after_sign():
    # Test cases that should match the original regex
    test_strings = [
        "This is a test. Hello",  # should match "Hello"
        "Is this correct? Yes",   # should match "Yes"
        "He said! Wow!",          # should match "Wow"
        "Are you okay? Yes, I'm fine.",  # should match "Yes"
        "Surprise! What now?"     # should match "What"
    ]
    
    # Test cases that should NOT match the original regex if the flags are changed incorrectly
    for test in test_strings:
        match = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test)
        assert match is not None, f"Expected a match for: '{test}' but got None"

    # Test a string that should not match
    no_match_strings = [
        "nothing to see here",    # no uppercase letter after a sign
        "Just regular text.",      # no uppercase letter after a sign
        "?why is it like this",    # should not match
        "hi!there",                # should not match
    ]
    
    for test in no_match_strings:
        match = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test)
        assert match is None, f"Expected no match for: '{test}' but got {match.group() if match else 'None'}"