from string_utils._regex import PRETTIFY_RE

def test__UPPERCASE_AFTER_SIGN():
    # Input strings that should trigger the UPPERCASE_AFTER_SIGN regex
    test_cases = [
        (". Hello", True),  # Should match
        ("! World", True),  # Should match
        ("? This is a test", True),  # Should match
        ("This is a test without a sign", False)  # Should not match
    ]

    # Check the regex against the test cases
    for text, expected in test_cases:
        if expected:
            # Check if the regex matches
            assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(text) is not None, f"Failed on input: {text}"
        else:
            # Check if the regex does not match
            assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(text) is None, f"Failed on input: {text}"

    print("All tests passed for UPPERCASE_AFTER_SIGN.")

# Call the test function
test__UPPERCASE_AFTER_SIGN()