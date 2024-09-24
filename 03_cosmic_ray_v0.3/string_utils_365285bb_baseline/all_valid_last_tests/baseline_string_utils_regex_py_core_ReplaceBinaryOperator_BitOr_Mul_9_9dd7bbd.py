import re
from string_utils._regex import PRETTIFY_RE

def test_UPPERCASE_AFTER_SIGN():
    # Test input strings 
    text_with_correct_behavior1 = "Hello. World"  # Should match
    text_with_correct_behavior2 = "Hi! There"      # Should match
    text_without_match = "No Match Here"            # Should not match
    text_lowercase_after_punct = "What is this? here"  # Should match original?
    text_uppercase_after_punct = "What is this? Here"  # Should match originally and mutant
    
    # Get the original regex
    original_regex = PRETTIFY_RE['UPPERCASE_AFTER_SIGN']
    
    # Validate original regex behavior
    assert original_regex.search(text_with_correct_behavior1) is not None, \
        "Expected a match with the correct regex for 'Hello. World'."

    assert original_regex.search(text_with_correct_behavior2) is not None, \
        "Expected a match with the correct regex for 'Hi! There'."

    assert original_regex.search(text_without_match) is None, \
        "Expected no match with the correct regex for 'No Match Here'."

    # This case should match based on the prior assumptions of regex behavior
    assert original_regex.search(text_lowercase_after_punct) is not None, \
        "Expected a match with the correct regex for 'What is this? here'."
    
    # Expect match here because uppercase letter follows punctuation
    assert original_regex.search(text_uppercase_after_punct) is not None, \
        "Expected a match with the correct regex for 'What is this? Here'."

    # Mutant testing
    mutant_regex = re.compile(r'([.?!]\s\w)', re.MULTILINE * re.UNICODE)

    # Test mutant regex matches
    assert mutant_regex.search(text_with_correct_behavior1) is not None, \
        "Mutant should match 'Hello. World' incorrectly (expected match)."
    
    assert mutant_regex.search(text_with_correct_behavior2) is not None, \
        "Mutant should match 'Hi! There' incorrectly (expected match)."

    assert mutant_regex.search(text_without_match) is None, \
        "Mutant should NOT match 'No Match Here' (expected no match)."
    
    # Duplicate check on the lowercase string to see if mutant fails as expected
    assert mutant_regex.search(text_lowercase_after_punct) is not None, \
        "Mutant should match 'What is this? here' unexpectedly (expected match)."
    
    # Expectation to also match where uppercase follows punctuation
    assert mutant_regex.search(text_uppercase_after_punct) is not None, \
        "Mutant should also match 'What is this? Here' incorrectly (expected match)."

# Execute the test function
test_UPPERCASE_AFTER_SIGN()