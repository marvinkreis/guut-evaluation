import re
from string_utils._regex import PRETTIFY_RE as correct_PRETTIFY_RE

def test__uppercase_after_sign():
    """Testing that the correct case matches and behaves as expected."""

    correct_pattern = correct_PRETTIFY_RE['UPPERCASE_AFTER_SIGN']
    
    # Sample inputs to verify behavior
    input_strings = [
        "Hello! World",            # Expect to match
        "This is fantastic? Awesome!",  # Expect to match
        "How about this: Amazing story.",  # Expect not to match (different punctuation)
        "No capital? ok!"          # Expect to match
    ]

    # Test each input string
    for input_string in input_strings:
        correct_matches = correct_pattern.findall(input_string)
        
        # Here we expect that it should find matches for cases where it should
        if "!" in input_string or "?" in input_string:
            assert len(correct_matches) > 0, f"Expected matches in correct implementation for '{input_string}'"
        else:
            assert len(correct_matches) == 0, f"Expected no matches in correct implementation for '{input_string}'"

# Run the test
test__uppercase_after_sign()