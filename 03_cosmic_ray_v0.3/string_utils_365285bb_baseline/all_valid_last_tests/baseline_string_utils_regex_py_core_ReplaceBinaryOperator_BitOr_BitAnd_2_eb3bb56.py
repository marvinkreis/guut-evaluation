from string_utils._regex import WORDS_COUNT_RE

def test_words_count_re():
    # Test strings that should match and not match
    test_cases = [
        ("Hello World!", True),          # Should match
        ("This is a test.", True),       # Should match
        ("", False),                      # Should not match
        ("   ", False),                  # Should not match
        ("Underscored_text", True),      # Should match
        ("Test1234", True),              # Should match
        ("Multiple   Spaces", True),     # Should match
        ("NoSymbolsHere123", True),      # Should match
        ("@#&*()", False),               # Should not match
        ("Word1, Word2; Word3.", True),  # Should match: with punctuation
        ("    Leading whitespace", True), # Should match: leading spaces
        ("Trailing whitespace    ", True) # Should match: trailing spaces
    ]

    for text, expected in test_cases:
        match = WORDS_COUNT_RE.search(text) is not None
        assert match == expected, f"Failed for '{text}': expected {expected}, got {match}"
        
    # Additional edge cases
    assert WORDS_COUNT_RE.search("A") is not None, "Failed for single letter 'A': expected match"
    assert WORDS_COUNT_RE.search("1 2 & 3") is not None, "Failed for '1 2 & 3': expected match"
    assert WORDS_COUNT_RE.search("\nNew line test") is not None, "Failed for newline case: expected match"
    assert WORDS_COUNT_RE.search("Tabs\t\tTest") is not None, "Failed for tabs: expected match"
    assert WORDS_COUNT_RE.search("   Text with spaces   ") is not None, "Failed for spaces around: expected match"

# Note: The assertions will raise an AssertionError if any test fails.