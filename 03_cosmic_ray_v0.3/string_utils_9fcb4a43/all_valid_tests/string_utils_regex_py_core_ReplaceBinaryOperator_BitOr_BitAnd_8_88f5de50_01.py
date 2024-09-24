from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE__SPACES_INSIDE():
    """Check that SPACES_INSIDE regex captures correctly; the mutant cannot match the same as the correct implementation."""
    # Example input that should match
    input_string = '  "Hello World"  '
    
    correct_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(input_string)
    assert len(correct_matches) > 0, "SPACES_INSIDE should match a quoted string."

# Test Output will follow.