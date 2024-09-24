from string_utils._regex import PRETTIFY_RE

def test_uppercase_after_sign():
    # Case where it should match (original regex)
    valid_case = "Hello! There is an uppercase after punctuation."
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(valid_case) is not None, "Expected match not found for valid input."

    # Case where it should NOT match (original regex)
    invalid_case = "Wow!this will not match."
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(invalid_case) is None, "Expected no match found for invalid input."

    # Case that mimics incorrect pattern matching in the mutant
    mutant_case = "Oops!Another sentence starts here."
    # In the original regex, this should NOT match because "Another" is lowercase.
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(mutant_case) is None, "Unexpected match for the correct implementation."

    # This case should match correctly
    match_case = "Check! This is the correct format."
    assert PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(match_case) is not None, "Expected match for uppercase after punctuation."

# Expected Results:
# 1. "Hello! There..." should match, indicating valid regex behavior.
# 2. "Wow!this will not match." should return None, indicating correct behavior.
# 3. "Oops!Another..." should also return None, indicating correct behavior (this is the point of error for the mutant).
# 4. "Check! This..." should match, confirming the regex works as intended.
