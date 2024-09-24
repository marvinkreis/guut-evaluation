import re
from string_utils._regex import PRETTIFY_RE

def test_uppercase_after_sign():
    # Input string that should match the regex in the original implementation
    test_string_valid = "Hello! This is a Test."
    test_string_invalid = "Hello This is a Test."
    
    # Test the original regex:
    # It should match 'T' in 'Test' after the '!'
    original_match_results = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string_valid)
    assert len(original_match_results) > 0, "Original regex should find an uppercase letter after punctuation"

    # Now let's create a test explicitly designed to fail with the mutant
    # The mutation affects how the regex operates, so when we expect a match for the valid string,
    # if the regex has been mutated, it should produce an unexpected result

    # Expect the mutant to not match an uppercase letter following punctuation
    mutant_match_results = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string_invalid)
    assert len(mutant_match_results) == 0, "Mutant should not capture any uppercase letter after punctuation in invalid scenario"

    # Let's include a deliberate check for the mutant by creating a string that would only match
    # when the OR operator works correctly. The mutant should fail at some point if not partially disabled.

    # We expect the original to successfully find uppercase letters appropriately
    assert original_match_results, "Original regex did not match successfully"

    # This will signal a failure if mutated version correctly matches or finds something unexpected
    result_for_uncertain_scenario = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall("Hello? Or not?")
    
    # Since the correct regex should find "O" after "?", it should yield 1
    assert len(result_for_uncertain_scenario) == 1, "Mutant should yield unexpected results for valid punctuation cases"