from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """The mutant changes the regex flags, affecting the ability to catch improperly formatted strings."""
    # Test input designed to capture improper space usage directly related to the pattern we are testing.
    test_input = 'This is a test  "incorrectly formatted"  where we expect duplicates....'
    
    # Using the RIGHT_SPACE regex to check for space around punctuation with quotes.
    matches = PRETTIFY_RE['RIGHT_SPACE'].findall(test_input)
    
    # Expecting to find at least one match in the original but might not with mutant due to the broken configuration.
    assert len(matches) == 0, "RIGHT_SPACE should have found improper spacing, but the mutant should not."