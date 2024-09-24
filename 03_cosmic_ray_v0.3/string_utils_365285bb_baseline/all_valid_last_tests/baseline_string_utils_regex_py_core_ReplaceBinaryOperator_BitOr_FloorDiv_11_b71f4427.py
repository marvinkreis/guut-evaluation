import re
from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Attempt to compile the regex to ensure it doesn't fail
    try:
        # Access the regex to ensure it compiles without errors
        compiled_regex = re.compile(PRETTIFY_RE['SPACES_INSIDE'])
    except re.error as e:
        raise AssertionError(f"Regex compilation failed: {e}")

    # Prepare test cases
    test_cases = [
        ('"Hello world" and (test)', True),   # Should match: Valid case
        ('"Quotes are here."', True),          # Should match: Valid case
        ('"This has an unmatched parenthesis(', False),  # Should NOT match: Invalid case
        ('No quotes or brackets here.', False), # Should NOT match
        ('(Valid parentheses)', True),         # Should match: Valid case
    ]

    for text, expected in test_cases:
        # Check matches using the compiled regex
        result = compiled_regex.search(text) is not None
        assert result == expected, f"String: {text} | Expected: {expected} | Got: {result}"

    # Adding edge cases with various structures
    edge_cases = [
        ('"   Leading and trailing spaces   "', True),  # Should match: Valid case with spaces
        ('(Text with parentheses)', True),                # Should match: Valid case with parentheses
        ('No special characters', False),                 # Should NOT match
        ('"Quotes without closing', False)                # Should NOT match: Invalid case
    ]

    for text, expected in edge_cases:
        result = compiled_regex.search(text) is not None
        assert result == expected, f"Edge Case - String: {text} | Expected: {expected} | Got: {result}"

# Execute the test function
test_PRETTIFY_RE()