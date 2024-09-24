from string_utils.validation import is_full_string

def test_is_full_string():
    # 1. Confirming whitespace input - must return False
    assert is_full_string('   ') == False, "Failed: Input with only whitespace should return False."

    # 2. Confirming empty string input - must return False
    assert is_full_string('') == False, "Failed: Empty input should return False."

    # 3. Valid input check - should return True
    assert is_full_string('hello') == True, "Failed: Valid string input should return True."

    # 4. Leading and trailing valid input - should return True
    assert is_full_string('   valid   ') == True, "Failed: Input with valid characters should return True."

    # 5. Checking for None input - should return False
    assert is_full_string(None) == False, "Failed: None input should return False."

    # 6. Single character valid input should return True.
    assert is_full_string('A') == True, "Failed: Input with one character should return True."

    # 7. Single space input (important edge case)
    assert is_full_string(' ') == False, "Failed: Single space input must return False."

    # 8. Input of all spaces (should catch mutant issue)
    assert is_full_string('    ') == False, "Failed: Multiple spaces input should return False."