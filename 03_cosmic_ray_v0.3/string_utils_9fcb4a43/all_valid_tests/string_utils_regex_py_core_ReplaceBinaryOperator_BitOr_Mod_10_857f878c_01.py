import re
from string_utils._regex import PRETTIFY_RE

def test__pretty_re():
    """Test to ensure that changes to the PRETTIFY_RE regex do not affect expected behavior."""
    
    # Test strings for checking regex patterns
    focused_test_strings = [
        "    Space before",       # Leading spaces, should match against 'DUPLICATES'
        "Space after    ",        # Trailing spaces 
        "Hello!!!",               # Special characters
        "Test_with_underscore",   # Underscore
    ]
    
    for name, pattern in PRETTIFY_RE.items():
        for test_string in focused_test_strings:
            match = re.match(pattern, test_string)
            if name == "DUPLICATES" and test_string == "    Space before":
                assert match is not None, f"{name} should match '{test_string}'"
            elif name == "DUPLICATES" and test_string == "Space after    ":
                assert match is None, f"{name} should NOT match '{test_string}'"
            elif name == "DUPLICATES" and test_string == "Hello!!!":
                assert match is None, f"{name} should NOT match '{test_string}'"
            elif name == "DUPLICATES" and test_string == "Test_with_underscore":
                assert match is None, f"{name} should NOT match '{test_string}'"

# Run the test
test__pretty_re()