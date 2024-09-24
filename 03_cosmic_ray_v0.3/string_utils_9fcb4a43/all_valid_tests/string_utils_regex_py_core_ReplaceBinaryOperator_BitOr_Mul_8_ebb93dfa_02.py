from string_utils._regex import PRETTIFY_RE

def test__multiple_regex():
    """ Test multiple PRETTIFY_RE patterns for matching spaces and patterns. """
    
    # Select patterns that address spaces more clearly.
    patterns_to_test = [
        PRETTIFY_RE['RIGHT_SPACE'],
        PRETTIFY_RE['LEFT_SPACE'],
        PRETTIFY_RE['SPACES_AROUND'],
        PRETTIFY_RE['SPACES_INSIDE'],
    ]
    
    test_strings = [
        "This is a test.",          # Should match spaces properly
        "A    B",                   # Should match due to multiple spaces
        "Match: (example)",         # Should match (valid parentheses)
        "Quote: \"hello world\"",   # Should match (valid quoted text)
        "Error: (   )",             # Should match (special case with spaces)
        "Multiple spaces  , here"   # Should match
    ]

    for pattern in patterns_to_test:
        print(f"\nTesting pattern: {pattern.pattern}")
        
        for s in test_strings:
            match_result = pattern.search(s) is not None
            print(f"Input: '{s}' => Match: {match_result}")

# Run the test
test__multiple_regex()