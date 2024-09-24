import re
from string_utils._regex import PRETTIFY_RE

def test__pretty_re():
    """Test to check the integrity of PRETTIFY_RE regex behavior."""
    
    # Test strings for checking regex patterns
    test_strings = [
        "    Space before",     # Leading spaces
        "Space after    ",      # Trailing spaces
        "Hello!!!",             # Special characters
        "(Hello)",              # Parentheses
        "Test_with_underscore"  # Underscore
    ]
    
    # Expected outcomes for the "DUPLICATES" regex
    expected_results = {
        "DUPLICATES": {
            "    Space before": True,  # Should match
            "Space after    ": False,  # Should NOT match
            "Hello!!!": False,          # Should NOT match
            "(Hello)": False,          # Should NOT match
            "Test_with_underscore": False,  # Should NOT match
        }
    }
    
    for name, pattern in PRETTIFY_RE.items():
        for test_string in test_strings:
            match = re.match(pattern, test_string)
            outcome = match is not None
            if name in expected_results:
                assert outcome == expected_results[name][test_string], f"{name} failed: '{test_string}'"

# Run the test
test__pretty_re()