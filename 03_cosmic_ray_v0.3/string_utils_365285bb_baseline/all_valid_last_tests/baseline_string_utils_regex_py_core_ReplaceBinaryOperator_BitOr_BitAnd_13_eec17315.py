from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test__no_letters_or_numbers_regex():
    # Define test cases and their expected match status
    test_cases = [
        ("abc123", None),     # Should not match (contains letters)
        ("abc!def", True),    # Should match because of '!' presence
        ("!@#$%^&*()", True),  # Should match (only special characters)
        ("___", True),        # Should match (only underscores)
        ("abc_", True),       # Should match because of '_' present
        ("12345", None),      # Should not match (only digits)
        ("!!!!", True),       # Should match (only special characters)
        ("    ", True),      # Should match (spaces)
        ("@#$%^&*()", True),  # Should match (only special characters)
        ("123_abc!", True),   # Should match because contains '!'
    ]

    for input_string, expected in test_cases:
        result = NO_LETTERS_OR_NUMBERS_RE.search(input_string)
        if expected is None:
            assert result is None, f"Failed for '{input_string}'. Expected None, got {result}."
        else:
            assert result is not None, f"Failed for '{input_string}'. Expected match, got None."

# Run the test case
test__no_letters_or_numbers_regex()