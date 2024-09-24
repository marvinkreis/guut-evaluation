import re
from string_utils._regex import PRETTIFY_RE as correct_PRETTIFY_RE

def test__uppercase_after_sign():
    """Test to expose the mutant's failure to match when it alters behavior."""
    
    correct_pattern = correct_PRETTIFY_RE['UPPERCASE_AFTER_SIGN']
    
    # Inputs designed to test different scenarios
    test_cases = [
        ("Hello! World", True),            # Correct regex should match, mutant might not
        ("This is fantastic? Awesome!", True),  # Correct should match
        ("Just-Check That! Fantastic.", True),  # Correct should match
        ("Now let's see how this works: Great.", False),  # Should not match
        ("Something random.", False),      # Should not match
        ("Where am I? What a view!", True) # Correct should match
    ]

    for input_string, should_match in test_cases:
        correct_matches = correct_pattern.findall(input_string)
        
        # Expect matches if should_match is True, else expect no matches
        if should_match:
            assert len(correct_matches) > 0, f"Expected matches in the correct implementation for '{input_string}'"
        else:
            assert len(correct_matches) == 0, f"Expected no matches in the correct implementation for '{input_string}'"

# Run the test
test__uppercase_after_sign()