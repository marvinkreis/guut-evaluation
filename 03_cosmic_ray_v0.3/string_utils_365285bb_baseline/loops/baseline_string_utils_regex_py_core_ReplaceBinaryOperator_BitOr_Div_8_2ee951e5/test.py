import re
from string_utils._regex import PRETTIFY_RE

def test_prettify_re():
    # The following string has extra spaces around punctuation that we want to capture
    test_string = 'This is a test string with  multiple     spaces, and   extra punctuation!'
    
    # Apply the regex to the test string to find duplicates
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # We expect to find multiple spaces, so we check the length of the matches
    assert len(matches) > 0  # Should pass with the correct code, as we should match the spaces.
    
    # The mutant modifies the regex to use / instead of |, which will lead to an error or incorrect behavior